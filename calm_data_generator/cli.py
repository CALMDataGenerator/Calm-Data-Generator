"""
CalmGenerator CLI - Command Line Interface
"""

import argparse
from pathlib import Path


def get_package_dir():
    """Get the calm_data_generator package directory."""
    return Path(__file__).parent


def get_tutorials_dir():
    """Get the tutorials directory."""
    # Tutorials are in the package's parent directory
    package_dir = get_package_dir()
    tutorials_dir = package_dir.parent / "tutorials"

    # If installed via pip, tutorials might be in package
    if not tutorials_dir.exists():
        tutorials_dir = package_dir / "tutorials"

    return tutorials_dir


def get_docs_dir():
    """Get the docs directory."""
    package_dir = get_package_dir()
    docs_dir = package_dir.parent / "docs"

    if not docs_dir.exists():
        docs_dir = package_dir / "docs"

    return docs_dir


def list_tutorials():
    """List all available tutorials."""
    tutorials_dir = get_tutorials_dir()

    if not tutorials_dir.exists():
        print("❌ Tutorials directory not found.")
        print(f"   Expected at: {tutorials_dir}")
        print("\n   Install from source or check GitHub:")
        print("   https://github.com/AlejandroBeldaFernandez/Calm-Data_Generator")
        return

    tutorials = sorted(tutorials_dir.glob("*.py"))

    if not tutorials:
        print("No tutorials found.")
        return

    print("\n📚 CalmGenerator Tutorials")
    print("=" * 50)

    for i, tutorial in enumerate(tutorials, 1):
        name = tutorial.stem
        # Read first docstring line
        with open(tutorial, "r") as f:
            content = f.read()
            # Find docstring
            if '"""' in content:
                start = content.find('"""') + 3
                end = content.find("\n", start)
                title = content[start:end].strip() if end != -1 else content[start:].strip()
            else:
                title = name.replace("_", " ").title()

        print(f"  {i}. {name}")
        print(f"     {title}")

    print("\n" + "=" * 50)
    print("Usage:")
    print("  calm_data_generator tutorials show <number>  - View tutorial")
    print("  calm_data_generator tutorials run <number>   - Run tutorial")
    print("  calm_data_generator tutorials path           - Show tutorials path")


def show_tutorial(number):
    """Show a specific tutorial."""
    tutorials_dir = get_tutorials_dir()
    tutorials = sorted(tutorials_dir.glob("*.py"))

    if not tutorials:
        print("No tutorials found.")
        return

    try:
        idx = int(number) - 1
        if 0 <= idx < len(tutorials):
            tutorial = tutorials[idx]
            print(f"\n📖 Tutorial: {tutorial.name}")
            print("=" * 60)
            with open(tutorial, "r") as f:
                print(f.read())
        else:
            print(f"❌ Invalid tutorial number. Choose 1-{len(tutorials)}")
    except ValueError:
        print("❌ Please provide a tutorial number (e.g., 1, 2, 3)")


def run_tutorial(number):
    """Run a specific tutorial."""
    import subprocess
    import sys

    tutorials_dir = get_tutorials_dir()
    tutorials = sorted(tutorials_dir.glob("*.py"))

    if not tutorials:
        print("No tutorials found.")
        return

    try:
        idx = int(number) - 1
        if 0 <= idx < len(tutorials):
            tutorial = tutorials[idx]
            print(f"\n Running: {tutorial.name}")
            print("=" * 60)
            # Run in a subprocess to keep the tutorial isolated from this process
            result = subprocess.run(
                [sys.executable, str(tutorial)],
                check=False,
            )
            if result.returncode != 0:
                print(f"Tutorial exited with code {result.returncode}")
        else:
            print(f"Invalid tutorial number. Choose 1-{len(tutorials)}")
    except ValueError:
        print("Please provide a tutorial number (e.g., 1, 2, 3)")
    except Exception as e:
        print(f"Error running tutorial: {e}")


def show_path():
    """Show the tutorials path."""
    tutorials_dir = get_tutorials_dir()
    docs_dir = get_docs_dir()

    print("\n📁 CalmGenerator Paths")
    print("=" * 50)
    print(f"  Tutorials: {tutorials_dir}")
    print(f"  Docs:      {docs_dir}")
    print(f"  Package:   {get_package_dir()}")


def show_docs():
    """Show available documentation."""
    docs_dir = get_docs_dir()

    if not docs_dir.exists():
        print("❌ Docs directory not found.")
        return

    docs = list(docs_dir.glob("*.md"))

    print("\n📖 CalmGenerator Documentation")
    print("=" * 50)

    for doc in docs:
        print(f"  - {doc.name}: {doc}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="calm_data_generator",
        description="CalmGenerator - Synthetic Data Generation Library",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Tutorials command
    tutorials_parser = subparsers.add_parser("tutorials", help="Access tutorials")
    tutorials_parser.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "show", "run", "path"],
        help="Action to perform",
    )
    tutorials_parser.add_argument(
        "number", nargs="?", help="Tutorial number (for show/run)"
    )

    # Docs command
    subparsers.add_parser("docs", help="Access documentation")

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "tutorials":
        if args.action == "list":
            list_tutorials()
        elif args.action == "show":
            if args.number:
                show_tutorial(args.number)
            else:
                print("❌ Please provide a tutorial number")
        elif args.action == "run":
            if args.number:
                run_tutorial(args.number)
            else:
                print("❌ Please provide a tutorial number")
        elif args.action == "path":
            show_path()
    elif args.command == "docs":
        show_docs()
    elif args.command == "version":
        from calm_data_generator import __version__

        print(f"calm_data_generator {__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
