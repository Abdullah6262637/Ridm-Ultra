from googletrans import Translator


class TranslationEngine:
    def __init__(self):
        self.translator = Translator()

    def to_english(self, text: str, src_lang: str = 'auto') -> str:
        """Translates input text to English for RAG search."""
        try:
            result = self.translator.translate(text, dest='en', src=src_lang)
            return result.text
        except Exception as e:
            print(f"[!] Translation Error (to EN): {e}")
            return text

    def to_target(self, text: str, dest_lang: str) -> str:
        """Translates the English output back to the target language."""
        try:
            result = self.translator.translate(text, dest=dest_lang, src='en')
            return result.text
        except Exception as e:
            print(f"[!] Translation Error (to {dest_lang}): {e}")
            return text
