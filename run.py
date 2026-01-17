# run.py

import time
from engine_runner import run_from_json
start = time.time()
#CONFIG = "test.json"  # <-- change juste ce nom de fichier
#CONFIG = "config_singletrack_demo.json"  # <-- change juste ce nom de fichier
#CONFIG = "test_offset_2.json"
#CONFIG = "test_singletrack_in_optimizer.json"
#CONFIG = "Train_funi_bus_sierre.json"
#CONFIG = "Train_funi_bus_sierre_4funi.json"
#CONFIG = "Train_funi_bus_sierre_funi_EHFalse.json"
#CONFIG = "Sierre_train_bus_BS1.json"
CONFIG = "config_R81_optimizer.json"
#CONFIG = "config_R81_test_scheduled.json"
run_from_json(CONFIG)

end = time.time()
elapsed = end - start

print("\n======================================")
print(f"Temps total d'exécution : {elapsed:.3f} secondes")
print("======================================\n")
