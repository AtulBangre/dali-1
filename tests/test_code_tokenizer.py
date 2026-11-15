import os
from dali.tokenizer import BytePairTokenizer

def verify_code_tokenizer(tokenizer_path: str = "checkpoints/code_bpe.json"):
    print("=" * 68)
    print("  STEP 5: CODE TOKENIZER ROUND-TRIP RECONSTRUCTION VERIFICATION")
    print("=" * 68)

    tokenizer = BytePairTokenizer()
    tokenizer.load(tokenizer_path)

    test_snippets = [
        "def solve(n: int) -> bool:\n    return n % 2 == 0 and n > 0",
        "export async function GET(req: Request) {\n    return NextResponse.json({ ok: true });\n}",
        "SELECT id, username FROM users WHERE status = 'active' ORDER BY created_at DESC;",
        ".container { display: flex; justify-content: center; align-items: center; }"
    ]

    all_passed = True
    for i, code in enumerate(test_snippets, 1):
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)

        # Normalize whitespace for comparison
        clean_orig = code.strip()
        clean_decoded = decoded.strip()
        is_exact = (clean_orig == clean_decoded)

        print(f"Snippet {i}: {len(tokens)} tokens | Exact Match: {is_exact}")
        if not is_exact:
            print(f"  [Original]: {repr(clean_orig)}")
            print(f"  [Decoded]:  {repr(clean_decoded)}")
            all_passed = False

    print("-" * 68)
    if all_passed:
        print("✅ Gate 2 PASSED: 100% Exact String Reconstruction Verified!")
    else:
        print("❌ Verification Failed: Divergence detected.")
    print("=" * 68)
    assert all_passed, "Code tokenizer reconstruction verification failed."

if __name__ == "__main__":
    verify_code_tokenizer()
