#!/usr/bin/env python3
"""Export the OpenAPI schema and generate TypeScript types.

Usage:
    python scripts/export_openapi.py          # Export OpenAPI JSON
    python scripts/export_openapi.py --check  # Check for drift (exit 1 if changed)

This is run by `make types` and should be idempotent on a clean tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Project roots
API_ROOT = Path(__file__).resolve().parent.parent / "api"
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def export_openapi() -> dict:
    """Import the app and export its OpenAPI schema."""
    import sys
    sys.path.insert(0, str(API_ROOT))
    
    from main import app
    
    # Export the OpenAPI schema
    return app.openapi()


def write_openapi_json(schema: dict, output_path: Path) -> None:
    """Write the OpenAPI schema to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"Exported OpenAPI schema to {output_path}")


def generate_typescript_types(openapi_path: Path, output_path: Path) -> bool:
    """Run openapi-typescript to generate TypeScript types.
    
    Returns True if successful, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["npx", "openapi-typescript", str(openapi_path), "-o", str(output_path)],
            cwd=str(WEB_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            print(f"Warning: openapi-typescript failed: {result.stderr}", file=sys.stderr)
            print("You may need to install it: cd web && npm install openapi-typescript", file=sys.stderr)
            return False
        
        print(f"Generated TypeScript types at {output_path}")
        return True
    except FileNotFoundError:
        print("Error: npx not found. Please install Node.js and npm.", file=sys.stderr)
        return False


def check_drift(openapi_path: Path, types_path: Path) -> bool:
    """Check if regenerating types would produce a diff.
    
    Returns True if there's no drift (types are up-to-date).
    """
    if not types_path.exists():
        print(f"Types file doesn't exist yet: {types_path}")
        print("Run without --check first to generate it.")
        return False
    
    # Read current types
    current_types = types_path.read_text(encoding="utf-8")
    
    # Generate new types to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        if not generate_typescript_types(openapi_path, temp_path):
            # If generation fails, assume drift to be safe
            return False
        
        new_types = temp_path.read_text(encoding="utf-8")
        
        if current_types != new_types:
            print("Types are out of date! Run `make types` to regenerate.")
            # Show a diff summary
            import difflib
            diff = difflib.unified_diff(
                current_types.splitlines(keepends=True),
                new_types.splitlines(keepends=True),
                fromfile=str(types_path),
                tofile=str(types_path) + " (regenerated)",
            )
            sys.stdout.writelines(diff)
            return False
        
        print("Types are up to date.")
        return True
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenAPI schema and generate TypeScript types")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for drift without writing files (exit 1 if changed)",
    )
    args = parser.parse_args()
    
    # Ensure web directory exists
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Paths
    openapi_json = WEB_ROOT / "openapi.json"
    types_ts = WEB_ROOT / "lib" / "api-types.ts"
    
    # Export OpenAPI schema
    schema = export_openapi()
    write_openapi_json(schema, openapi_json)
    
    if args.check:
        # Check for drift
        if check_drift(openapi_json, types_ts):
            print("✓ Types are up to date")
            return 0
        else:
            print("✗ Types are out of date")
            return 1
    else:
        # Generate types
        if generate_typescript_types(openapi_json, types_ts):
            print("✓ Types generated successfully")
            return 0
        else:
            # Generate the JSON at least, types are optional
            print("⚠ OpenAPI JSON exported, but TypeScript types generation failed")
            print("  Install openapi-typescript: cd web && npm install openapi-typescript")
            return 0  # Don't fail, JSON is still useful


if __name__ == "__main__":
    sys.exit(main())
