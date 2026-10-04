
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler 
from langchain.agents import create_agent
from langchain.tools import tool
from pathlib import Path
import sys
import subprocess

load_dotenv()

def get_project_root() -> Path:
    if len(sys.argv) != 2:
        print("Usage: python agent.py <project-directory>")
        sys.exit(1)

    project_path = Path(sys.argv[1]).expanduser().resolve()

    if not project_path.exists():
        print(f"Error: Project directory does not exist: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path is not a directory: {project_path}")
        sys.exit(1)

    return project_path

PROJECT_ROOT = get_project_root()
print(f"Dev Assistant Workspace: {PROJECT_ROOT}")

THREAD_ID="thread-coding-assistant-107"
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

Before performing a coding task, use analyze_repo when you need
to understand the project's language, framework, package manager,
or testing framework.

Use analyze_repo especially when:
- the project structure is unfamiliar
- you need to determine how tests should be run
- you need to determine the appropriate build/lint command
- you need to understand the project's technology stack

Do not call analyze_repo repeatedly if the repository context
is already known.

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

@tool
def search_code(query: str) -> str:
    """
    Search the project source code for a text pattern.

    Use this tool to find where a function, class, variable,
    error message, or other piece of code is used.
    Returns matching files and line numbers.
    """
    if not query.strip():
        return "Search query cannot be empty."

    results = []

    for file_path in PROJECT_ROOT.rglob("*"):
        # Ignore directories/files we don't want to search
        if any(part in IGNORED_NAMES for part in file_path.parts):
            continue

        if not file_path.is_file():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        for line_number, line in enumerate(content.splitlines(), start=1):
            if query.lower() in line.lower():
                relative_path = file_path.relative_to(PROJECT_ROOT)

                results.append(
                    f"{relative_path}:{line_number}: {line.strip()}"
                )

    if not results:
        return f"No matches found for: {query}"

    return "\n".join(results)

@tool
def analyze_repo() -> str:
    """
    Analyze the project repository and identify its language,
    framework, package manager, and likely test commands.
    Use this before making decisions about how to inspect,
    test, or modify the project.
    """

    files = {
        file.name
        for file in PROJECT_ROOT.rglob("*")
        if file.is_file()
        and not any(part in IGNORED_NAMES for part in file.parts)
    }

    result = []

    # Language detection
    languages = []

    if any(file.endswith(".py") for file in files):
        languages.append("Python")

    if any(
        file.endswith((".js", ".jsx", ".ts", ".tsx"))
        for file in files
    ):
        languages.append("JavaScript/TypeScript")

    if languages:
        result.append(f"Languages: {', '.join(languages)}")

    # Package manager / project files
    if "package.json" in files:
        package_manager = "npm"

        if "pnpm-lock.yaml" in files:
            package_manager = "pnpm"
        elif "yarn.lock" in files:
            package_manager = "yarn"
        elif "package-lock.json" in files:
            package_manager = "npm"

        result.append(f"Package manager: {package_manager}")

    if "requirements.txt" in files:
        result.append("Python dependency file: requirements.txt")

    if "pyproject.toml" in files:
        result.append("Python project configuration: pyproject.toml")

    # Test framework detection
    if (
        "pytest.ini" in files
        or "pytest.ini" in files
        or "pyproject.toml" in files
    ):
        result.append("Possible Python test framework: pytest")

    if "package.json" in files:
        result.append(
            "JavaScript/TypeScript tests may be configured in package.json"
        )

    if "jest.config.js" in files or "jest.config.ts" in files:
        result.append("Test framework: Jest")

    if "vitest.config.ts" in files:
        result.append("Test framework: Vitest")

    if not result:
        return "Could not determine project information."

    return "\n".join(result)


def main():
    agent = create_agent(
        model=OPEN_AI_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            analyze_repo,
            list_files,
            read_file,
            edit_file,
            run_command,
            search_code
        ],
    )

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        },
        "callbacks": [langfuse_handler],
    }

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
