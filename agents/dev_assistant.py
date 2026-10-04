
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler 
from langchain.agents import create_agent
from langchain.tools import tool
from pathlib import Path
import subprocess

load_dotenv()

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent / "sample-app"
).resolve()

THREAD_ID="thread-coding-assistant-106"
OPEN_AI_MODEL="gpt-4o-mini"
MAX_ITERATIONS = 10

IGNORED_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    "dist",
    "build",
}

SYSTEM_PROMPT = """
You are a coding agent working on a local software project.

The project workspace is provided through your tools.
Always inspect the project before deciding how to test it.

When asked to fix a bug or failing test:

1. Use list_files to inspect the project.
2. Identify the programming language and testing framework from the files.
3. Read the relevant source and test files.
4. Run the appropriate test command using run_command.
5. Identify the actual failure from the command output.
6. Fix the code using edit_file.
7. Run the tests again.
8. Repeat until the tests pass or you cannot proceed.
9. Never claim a test passed unless you actually ran it and received a successful exit code.

IMPORTANT:
- Do not assume the project uses npm, pytest, unittest, or any other framework.
- Determine the appropriate test command by inspecting the project.
- Do not install dependencies unless explicitly asked.
- Do not modify files when the user asks only to inspect or run tests.
- Do not modify unrelated files.
- Never claim success based on assumptions.
"""

langfuse_handler = CallbackHandler()

# Stop agent from accessing file outside project.
def safe_path(file_path: str) -> Path:
    """
    Resolve a path and make sure it stays inside PROJECT_ROOT.
    Prevents paths such as ../../secrets.txt.
    """
    path = (PROJECT_ROOT / file_path).resolve()

    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Access denied: path is outside the project directory.")

    return path

@tool
def list_files(path: str = ".") -> str:
    """
    List files and directories inside the project.

    Use this tool to explore the codebase and discover relevant files.
    Path should be relative to the project root.
    """
    directory = safe_path(path)

    if not directory.exists():
        return f"Directory does not exist: {path}"

    if not directory.is_dir():
        return f"Not a directory: {path}"

    items = []

    # yield everything inside directory alphabetically
    for item in sorted(directory.iterdir()):
        if item.name in IGNORED_NAMES:
            continue
        
        relative_path = item.relative_to(PROJECT_ROOT) # Work relative to project directories only

        if item.is_dir():
            items.append(f"[DIR]  {relative_path}/")
        else:
            items.append(f"[FILE] {relative_path}")

    if not items:
        return "Directory is empty."

    return "\n".join(items)

@tool
def read_file(path: str) -> str:
    """
    Read the contents of a source code file.

    Use this tool when you need to understand existing code before making changes.
    Path must be relative to the project root.
    """
    file_path = safe_path(path)

    if not file_path.exists():
        return f"File does not exist: {path}"

    if not file_path.is_file():
        return f"Not a file: {path}"

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Unable to read {path}: file is not a UTF-8 text file."

@tool
def edit_file(path: str, content: str) -> str:
    """
    Replace the entire contents of a source code file.

    Use this tool only after reading the file and understanding the existing code.
    Human approval is required before the file is modified.
    """
    file_path = safe_path(path)

    if not file_path.exists():
        return f"File does not exist: {path}"

    if not file_path.is_file():
        return f"Not a file: {path}"

    print("\n" + "=" * 60)
    print("⚠️  AGENT WANTS TO MODIFY A FILE")
    print("=" * 60)
    print(f"File: {path}")
    print("\nApprove this change? [y/n]: ", end="")

    approval = input().strip().lower()

    if approval not in ("y", "yes"):
        return "EDIT DENIED by user. Do not attempt this edit again."

    try:
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully updated {path}"

    except Exception as e:
        return f"Failed to update {path}: {str(e)}"

@tool
def run_command(command: str) -> str:
    """
    Run a command inside the project directory.
    Use this to run tests, linting, builds, or other project checks.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        return (
            f"COMMAND: {command}\n"
            f"EXIT_CODE: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    except subprocess.TimeoutExpired:
        return (
            f"COMMAND: {command}\n"
            f"ERROR: Command timed out after 60 seconds."
        )

    except Exception as e:
        return (
            f"COMMAND: {command}\n"
            f"ERROR: {str(e)}"
        )


def main():
    agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            list_files,
            read_file,
            edit_file,
            run_command
        ],
    )

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        },
        "callbacks": [langfuse_handler],
    }

    print("AI Developer Assistant")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                },
                config={
                    **config,
                    "recursion_limit": MAX_ITERATIONS,
                }
            )

            print(
                f"\nAgent: "
                f"{response['messages'][-1].content}\n"
            )

        except Exception as e:
            print(f"\nAgent error: {e}\n")   


if __name__ == "__main__":
    main()
