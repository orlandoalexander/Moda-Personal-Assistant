import sys, os
sys.path.append(os.path.abspath(".."))
print(sys.path.append(os.path.abspath("..")))

try:
  import google.colab  # so I only do it when I'm on google colab
  sys.path.insert(0, "/content/preproc")
except:
  pass

import aa
