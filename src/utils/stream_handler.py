import sys
import time
from transformers import TextIteratorStreamer


class StreamHandler(TextIteratorStreamer):
    def __init__(self, tokenizer, skip_prompt=True, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.token_count = 0
        self.start_time = None
        self.generated_text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.start_time is None:
            self.start_time = time.time()

        self.token_count += len(self.tokenizer.encode(text))
        self.generated_text += text

        sys.stdout.write(text)
        sys.stdout.flush()

        if stream_end:
            elapsed_time = time.time() - self.start_time
            tokens_per_second = self.token_count / elapsed_time if elapsed_time > 0 else 0
            sys.stdout.write(
                f"\n\nGeneration completed. Total tokens: {self.token_count} | Speed: {tokens_per_second:.2f} tokens/sec\n")
            sys.stdout.flush()
