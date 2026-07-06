"""Re-export from baselines.herzig.herzig_faithful."""
import sys
from pathlib import Path
_replication_dir = str(Path(__file__).resolve().parent.parent.parent)
if _replication_dir not in sys.path:
    sys.path.insert(0, _replication_dir)

from baselines.herzig.herzig_faithful import *  # noqa: F401,F403,E402
from baselines.herzig.herzig_faithful import predict_herzig_faithful  # noqa: F811,E402
