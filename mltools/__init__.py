
# import needed utility functions into main namespace
from .utils import *

# import "base" (abstract) classifier and regressor classes
from .base import *

# import "plot" functions into main namespace
from .plot import *

# import feature transforms etc into sub-namespace
from . import transforms 

# import classifiers into sub-namespaces
from . import knn 

from . import cluster


