"""Constantes compartidas entre el entrenamiento y la API (sin dependencias pesadas)."""

# Clases de Fashion-MNIST (en español)
CLASSES = [
    "Polera",        # 0 T-shirt/top
    "Pantalón",      # 1 Trouser
    "Poleron",       # 2 Pullover
    "Vestido",       # 3 Dress
    "Abrigo",        # 4 Coat
    "Sandalia",      # 5 Sandal
    "Camisa",        # 6 Shirt
    "Zapatilla",     # 7 Sneaker
    "Bolso",         # 8 Bag
    "Bota",          # 9 Ankle boot
]

# Media y desviación estándar de los píxeles de Fashion-MNIST (para normalizar)
MEAN = 0.2860
STD = 0.3530

IMG_SIZE = 28
