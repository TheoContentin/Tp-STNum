import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Pour lire les fichiers

cars = pd.read_table("cars.txt",
                        sep=";",
                        decimal=".")

