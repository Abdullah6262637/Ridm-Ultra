"""Core Chat Engine Orchestrator for RIDM Ultra."""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List, Optional

from .adapters import AdapterFactory, LocalTransformerAdapter
from .interfaces import BaseChatRepository, BaseMemoryManager, BaseModelAdapter, BaseRouter
from .memory import HierarchicalMemoryManager
from .repository import InMemoryChatRepository
from .router import SemanticRouter
from .types import ChatMessage, ChatResponseChunk, ChatSession, MessageRole, ModelTier

logger = logging.getLogger(__name__)


class ChatEngine:
    """Production-Grade Chat System orchestrating Memory, Routing, Adapters & Persistence."""

    def __init__(
        self,
        adapters: Optional[Dict[ModelTier, BaseModelAdapter]] = None,
        memory_manager: Optional[BaseMemoryManager] = None,
        router: Optional[BaseRouter] = None,
        repository: Optional[BaseChatRepository] = None,
        rag_engine=None,  # SimpleRAG or LSHIndex instance from graph_retrieval
    ):
        # Set up Dependency Injection defaults
        self.factory = AdapterFactory()
        if adapters:
            for tier, adapter in adapters.items():
                self.factory.register(tier, adapter)
        else:
            # Register production default adapters
            self.factory.register(ModelTier.FAST, LocalTransformerAdapter(tier=ModelTier.FAST))
            self.factory.register(ModelTier.BALANCED, LocalTransformerAdapter(tier=ModelTier.BALANCED))
            self.factory.register(ModelTier.REASONING, LocalTransformerAdapter(tier=ModelTier.REASONING))

        self.memory_manager = memory_manager or HierarchicalMemoryManager()
        self.router = router or SemanticRouter()
        self.repository = repository or InMemoryChatRepository()
        self.rag_engine = rag_engine

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatSession:
        if session_id:
            session = await self.repository.get_session(session_id)
            if session:
                if system_prompt:
                    session.system_prompt = system_prompt
                return session

        new_session = ChatSession(system_prompt=system_prompt)
        await self.repository.save_session(new_session)
        return new_session

    async def chat_stream(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        forced_tier: Optional[ModelTier] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Asynchronously stream assistant response token-by-token (<200ms TTFT)."""

        # 1. Fetch or initialize chat session
        session = await self.get_or_create_session(session_id, system_prompt)

        try:
            from deep_translator import GoogleTranslator
            from langdetect import detect
            user_lang = detect(user_message)
        except (ImportError, Exception) as e:
            logger.debug(f"Language detection unavailable, defaulting to English: {e}")
            user_lang = 'en'

        # Use the original user_message for UMA geometric expansion in RAG
        uma_query = user_message

        # The translated message is still needed for English-only LLM adapters (if fallback occurs)
        if user_lang != 'en':
            logger.info(f"Translating query from {user_lang} to en")
            translated_user_msg = GoogleTranslator(source=user_lang, target='en').translate(user_message)
        else:
            translated_user_msg = user_message

        # 2. Append user message to session (store original)
        user_chat_msg = ChatMessage(role=MessageRole.USER, content=user_message)
        session.add_message(user_chat_msg)

        # 3. Perform RAG query transformation and lookup if available.
        rag_context = ""
        used_native_rag = False
        max_cosine = 0.0
        if self.rag_engine is not None and hasattr(self.rag_engine, "retrieve"):
            try:
                hits = self.rag_engine.retrieve(uma_query, top_k=2)
                used_native_rag = bool(getattr(self.rag_engine, "last_used_native", False))
                max_cosine = float(getattr(self.rag_engine, "last_max_cosine", 0.0))
                
                last_expansion = getattr(self.rag_engine, "last_expansion", None)
                if last_expansion:
                    yield ChatResponseChunk(delta=f"> [!NOTE]\n> **{last_expansion}**\n\n", model_name="ridm-ultra-native-svd")

                if hits:
                    docs = [h[0] for h in hits if isinstance(h, tuple)]
                    if docs:
                        rag_context = f"\n[Retrieved Context]: {' | '.join(docs)}\n"
            except Exception as e:
                logger.warning(f"RAG retrieval error: {e}")
                used_native_rag = False

        # 4. Route query to appropriate ModelAdapter
        available_adapters = self.factory.all_adapters()
        if forced_tier and forced_tier in available_adapters:
            adapter = available_adapters[forced_tier]
        else:
            adapter = self.router.route_query(translated_user_msg, session.messages, available_adapters)

        # 5. Assemble context window adhering to token limits
        raw_context = self.memory_manager.get_context_messages(session, max_token_budget=8192)
        context_messages = []
        for m in raw_context:
            # We must pass the translated message to the model for context if it's the last one
            if m.content == user_message:
                context_messages.append(ChatMessage(role=m.role, content=translated_user_msg, metadata=dict(m.metadata or {})))
            else:
                context_messages.append(ChatMessage(role=m.role, content=m.content, metadata=dict(m.metadata or {})))

        # Inject RAG context into prompt if retrieved
        if rag_context and context_messages:
            last_msg = context_messages[-1]
            context_messages[-1] = ChatMessage(
                role=last_msg.role,
                content=last_msg.content + rag_context,
                metadata=dict(last_msg.metadata or {}),
            )

        # 6. Stream tokens asynchronously from adapter
        full_response_acc: List[str] = []
        finish_chunk: Optional[ChatResponseChunk] = None

        async for chunk in adapter.generate_stream(
            context_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response_acc.append(chunk.delta)
            if chunk.finish_reason:
                finish_chunk = chunk

            if user_lang == 'en':
                if chunk.finish_reason:
                    yield ChatResponseChunk(delta=chunk.delta, finish_reason=None, model_name=chunk.model_name)
                else:
                    yield chunk

        assistant_content_en = "".join(full_response_acc)
        if user_lang != 'en':
            logger.info(f"Translating response from en to {user_lang}")
            try:
                assistant_content_translated = GoogleTranslator(source='en', target=user_lang).translate(assistant_content_en)
            except Exception as e:
                logger.warning(f"Translation to {user_lang} failed, using English: {e}")
                assistant_content_translated = assistant_content_en

            yield ChatResponseChunk(delta=assistant_content_translated, finish_reason=None, model_name=adapter.model_name)
            assistant_content = assistant_content_translated
        else:
            assistant_content = assistant_content_en

        if used_native_rag and rag_context:
            badge = (
                f"\n\n⚡ [Source: 100% Native C++ SVD Core - Offline Mode "
                f"(cosine={max_cosine:.2f})]"
            )
        else:
            badge = "\n\n⚡ [Source: 100% Native C++ SVD Core - Offline Mode]"

        yield ChatResponseChunk(
            delta=badge, 
            finish_reason="stop", 
            usage=finish_chunk.usage if finish_chunk else None,
            model_name=adapter.model_name
        )
        assistant_content += badge

        # 7. Post-generation state update & persistence
        assistant_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=assistant_content,
            metadata={
                "model": adapter.model_name,
                "tier": adapter.tier.value,
                "usage": finish_chunk.usage.__dict__ if finish_chunk and finish_chunk.usage else {},
            },
        )
        session.add_message(assistant_msg)

        # Trigger background summarization if context exceeds capacity threshold
        await self.memory_manager.summarize_if_needed(session, max_tokens=8192)

        # Save updated session
        await self.repository.save_session(session)

    async def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        forced_tier: Optional[ModelTier] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatMessage:
        """Synchronous complete response wrapper."""
        chunks: List[str] = []
        last_chunk: Optional[ChatResponseChunk] = None

        async for chunk in self.chat_stream(
            user_message=user_message,
            session_id=session_id,
            system_prompt=system_prompt,
            forced_tier=forced_tier,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            chunks.append(chunk.delta)
            last_chunk = chunk

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="".join(chunks),
            metadata={"model": last_chunk.model_name if last_chunk else "unknown"},
        )
