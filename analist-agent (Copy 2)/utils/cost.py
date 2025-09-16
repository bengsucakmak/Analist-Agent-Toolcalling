import math

class CostTracker:
    def __init__(self, in_price_per_1k=0.2, out_price_per_1k=0.6):
        self.in_price = in_price_per_1k
        self.out_price = out_price_per_1k
        self.input_tokens = 0
        self.output_tokens = 0

    @staticmethod
    def est_tokens(text: str) -> int:
        # kaba tahmin: 4 karakter ≈ 1 token
        return max(1, math.ceil(len(text)/4))

    def add_call(self, prompt: str, output: str):
        it = self.est_tokens(prompt)
        ot = self.est_tokens(output)
        self.input_tokens += it
        self.output_tokens += ot

    def usd(self):
        return (self.input_tokens/1000.0)*self.in_price + (self.output_tokens/1000.0)*self.out_price

    def to_dict(self):
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "usd": round(self.usd(), 6)}