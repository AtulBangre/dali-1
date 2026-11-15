import os

def compile_full_stack_master(output_path: str = "knowledge_base/FULL_STACK_MASTER_CORPUS.txt"):
    dirs = [
        "knowledge_base/00_html5_standards",
        "knowledge_base/01_tailwind_ui",
        "knowledge_base/02_react_nextjs",
        "knowledge_base/03_mern_backend",
        "knowledge_base/04_fastapi_postgres",
        "knowledge_base/05_docker_devops"
    ]
    all_blocks = []
    seen = set()

    for d in dirs:
        if not os.path.exists(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".txt"):
                fpath = os.path.join(d, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                blocks = [b.strip() for b in content.split("<|end_block|>") if b.strip()]
                for b in blocks:
                    norm = b[:150].lower()
                    if norm not in seen:
                        seen.add(norm)
                        all_blocks.append(b)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n\n<|end_block|>\n\n".join(all_blocks))

    print(f"✓ Master Full-Stack Corpus compiled with {len(all_blocks)} official tech blocks into {output_path}")
    return len(all_blocks)

if __name__ == "__main__":
    compile_full_stack_master()
