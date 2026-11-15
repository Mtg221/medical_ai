"""Export FastAPI OpenAPI 3.1 schema and generate TypeScript types via openapi-typescript."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from medresearch_platform.interfaces.api import create_app


def main() -> None:
    app = create_app()
    schema = app.openapi()
    openapi_path = Path("frontend/openapi.json")
    openapi_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    out_ts = Path("frontend/src/types/openapi.d.ts")
    subprocess.run(
        ["npx", "--prefix", "frontend", "openapi-typescript", str(openapi_path), "-o", str(out_ts)],
        check=True,
    )
    print(f"Generated {out_ts} from {openapi_path} ({len(schema.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
