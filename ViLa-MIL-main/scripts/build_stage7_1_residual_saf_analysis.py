from pathlib import Path
import runpy


ROOT_DIR = Path("/xiangmu/ViLMIL/ViLa-MIL-main")
TARGET = ROOT_DIR / "scripts" / "analysis" / "build_stage7_1_residual_saf_analysis.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
