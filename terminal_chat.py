"""RIDM-ULTRA Terminal Chat — interaktif terminal sohbet arayüzü.

Kullanım:
    python terminal_chat.py [--session SESSION_ID]

Komutlar:
    /quit, /exit, /q   — Çıkış
    /clear              — Oturum geçmişini temizle
    /history            — Mesaj geçmişini göster
    /session            — Mevcut oturum ID'sini göster
    /help               — Komut listesi
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import argparse
from typing import Optional

# Proje kökünü sys.path'e ekle (düz modül importları için)
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ridm_ultra.chat.engine import ChatEngine
from ridm_ultra.chat.adapters import LocalTransformerAdapter, AdapterFactory
from ridm_ultra.chat.repository import SQLiteChatRepository
from ridm_ultra.chat.types import ModelTier, MessageRole

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("terminal_chat")

# ─── ANSI Renk Kodları ───────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

BANNER = f"""
{CYAN}{BOLD}+==============================================================+
|              RIDM-ULTRA v6  -  Terminal Chat                 |
|         100% Offline  -  Native C++ SVD Core                 |
+==============================================================+{RESET}

{DIM}Komutlar: /help  -  Cikis: /quit{RESET}
"""

HELP_TEXT = f"""
{YELLOW}{BOLD}Komutlar:{RESET}
  {GREEN}/quit, /exit, /q{RESET}   Çıkış
  {GREEN}/clear{RESET}             Oturum geçmişini temizle
  {GREEN}/history{RESET}           Mesaj geçmişini göster
  {GREEN}/session{RESET}           Mevcut oturum ID'sini göster
  {GREEN}/help{RESET}              Bu yardım mesajını göster
"""


def _build_engine() -> ChatEngine:
    """ChatEngine'i NativeDecoder tabanlı adapter ile oluşturur."""
    adapter = LocalTransformerAdapter(
        model_name="ridm-ultra-native-svd",
        tier=ModelTier.FAST,
    )
    factory = AdapterFactory()
    factory.register(ModelTier.FAST, adapter)
    factory.register(ModelTier.BALANCED, adapter)
    factory.register(ModelTier.REASONING, adapter)

    db_path = os.path.join(_project_root, "artifacts", "chat_sessions.db")
    repository = SQLiteChatRepository(db_path=db_path)

    rag_engine = _build_rag_engine()

    engine = ChatEngine(
        adapters={
            ModelTier.FAST: adapter,
            ModelTier.BALANCED: adapter,
            ModelTier.REASONING: adapter,
        },
        repository=repository,
        rag_engine=rag_engine,
    )
    return engine


def _build_rag_engine():
    """Eğer FineWeb RAG verisi varsa SimpleRAG döndürür, yoksa None."""
    try:
        from ridm_ultra.llm.native_decoder import NativeDecoder
        decoder = NativeDecoder()
        if decoder.word_emb is not None and len(decoder.vocab) > 0:
            from graph_retrieval import SimpleRAG

            class _NativeRAGBridge:
                """NativeDecoder embedding'lerini SimpleRAG arayüzüne bağlar."""
                def __init__(self, decoder_inst):
                    self._dec = decoder_inst
                    self.last_used_native = True
                    self.last_max_cosine = 0.0
                    self.last_expansion = None

                def retrieve(self, query: str, top_k: int = 2):
                    tokens = query.lower().split()
                    known = [t for t in tokens if t in self._dec.word2idx]
                    if not known:
                        return []
                    ids = [self._dec.word2idx[w] for w in known]
                    q_vec = self._dec.word_emb[ids].mean(axis=0)
                    import numpy as np
                    norms = np.linalg.norm(self._dec.word_emb, axis=1, keepdims=True)
                    safe_norms = np.where(norms < 1e-8, 1.0, norms)
                    cosines = (self._dec.word_emb @ q_vec) / (safe_norms.squeeze() * (np.linalg.norm(q_vec) + 1e-8))
                    best_ids = np.argsort(cosines)[::-1][:top_k * 10]
                    results = []
                    seen = set(known)
                    for wid in best_ids:
                        w = self._dec.vocab[wid]
                        if w not in seen and len(w) > 1:
                            results.append((w, float(cosines[wid])))
                            seen.add(w)
                            if len(results) >= top_k:
                                break
                    if results:
                        self.last_max_cosine = results[0][1]
                    return results

            return _NativeRAGBridge(decoder)
    except Exception as e:
        logger.debug(f"RAG engine başlatılamadı: {e}")
    return None


async def _stream_response(engine: ChatEngine, user_msg: str, session_id: str) -> None:
    """ChatEngine'den gelen chunk'ları anlık olarak terminale basar."""
    sys.stdout.write(f"\n{MAGENTA}{BOLD}RIDM > {RESET}")
    sys.stdout.flush()

    token_count = 0
    async for chunk in engine.chat_stream(
        user_message=user_msg,
        session_id=session_id,
        temperature=0.7,
        max_tokens=1024,
    ):
        text = chunk.delta
        if text:
            # Badge satırlarını daha dim göster
            if text.startswith("⚡"):
                sys.stdout.write(f"{DIM}{text}{RESET}")
            elif text.startswith(">"):
                sys.stdout.write(f"{BLUE}{text}{RESET}")
            else:
                sys.stdout.write(text)
            sys.stdout.flush()
            token_count += 1

    sys.stdout.write("\n")
    sys.stdout.flush()


def _show_history(engine: ChatEngine, session_id: str) -> None:
    """Oturumdaki mesaj geçmişini gösterir."""
    loop = asyncio.get_event_loop()
    session = loop.run_until_complete(engine.repository.get_session(session_id))
    if not session or not session.messages:
        print(f"{DIM}(Gecmis bos){RESET}")
        return

    print(f"\n{YELLOW}{BOLD}-- Oturum Gecmisi --{RESET}")
    for msg in session.messages:
        role = msg.role.value.upper()
        if msg.role == MessageRole.USER:
            color = GREEN
        elif msg.role == MessageRole.ASSISTANT:
            color = MAGENTA
        else:
            color = DIM
        content_preview = msg.content[:120] + ("..." if len(msg.content) > 120 else "")
        print(f"  {color}{BOLD}{role:10s}{RESET} {content_preview}")
    print()


async def _clear_session(engine: ChatEngine, session_id: str) -> str:
    """Mevcut oturumu siler, yeni oturum başlatır."""
    await engine.repository.delete_session(session_id)
    new_session = await engine.get_or_create_session()
    return new_session.session_id


async def _run_chat(engine: ChatEngine, initial_session_id: Optional[str] = None) -> None:
    """Ana REPL döngüsü."""
    session = await engine.get_or_create_session(session_id=initial_session_id)
    session_id = session.session_id

    print(BANNER)
    print(f"{DIM}Oturum: {session_id[:8]}...{RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}Sen > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Gorusuruz!{RESET}")
            break

        if not user_input:
            continue

        # Slash komutları
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print(f"{DIM}Gorusuruz!{RESET}")
            break
        if cmd == "/help":
            print(HELP_TEXT)
            continue
        if cmd == "/session":
            print(f"{DIM}Oturum ID: {session_id}{RESET}")
            continue
        if cmd == "/history":
            _show_history(engine, session_id)
            continue
        if cmd == "/clear":
            session_id = await _clear_session(engine, session_id)
            print(f"{YELLOW}Oturum temizlendi. Yeni oturum: {session_id[:8]}...{RESET}\n")
            continue

        await _stream_response(engine, user_input, session_id)


def main():
    parser = argparse.ArgumentParser(description="RIDM-ULTRA Terminal Chat")
    parser.add_argument("--session", type=str, default=None,
                        help="Mevcut bir oturum ID'si ile devam et.")
    args = parser.parse_args()

    try:
        engine = _build_engine()
    except FileNotFoundError as e:
        print(f"{RED}{BOLD}HATA:{RESET} {e}")
        print(f"{DIM}Önce embedding artifact'lerini oluşturun.{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}{BOLD}HATA:{RESET} Motor başlatma hatası: {e}")
        sys.exit(1)

    print(f"{DIM}Motor yüklendi.{RESET}")

    asyncio.run(_run_chat(engine, initial_session_id=args.session))


if __name__ == "__main__":
    main()
