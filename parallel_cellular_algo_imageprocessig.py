# --- SIMPLE PARALLEL CELLULAR ALGORITHM (NO IMPORTS) ---

# Example grayscale "image" (each pixel: 0–255)
image = [
    [ 20,  30,  40,  30,  20],
    [ 30, 100, 150, 100,  30],
    [ 40, 150, 255, 150,  40],
    [ 30, 100, 150, 100,  30],
    [ 20,  30,  40,  30,  20]
]

# Parameters
iterations = 5      # number of iterations
alpha = 0.4         # blending factor (how much neighbor average influences pixel)
height = len(image)
width = len(image[0])

def get_neighbors(img, x, y):
    """Return 3x3 neighborhood around (x,y) with edge handling."""
    neighbors = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if 0 <= i < height and 0 <= j < width:
                neighbors.append(img[i][j])
    return neighbors

def parallel_update(img):
    """Perform one parallel update step."""
    new_img = [[0]*width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            neighbors = get_neighbors(img, i, j)
            mean_value = sum(neighbors) / len(neighbors)
            new_img[i][j] = int((1 - alpha) * img[i][j] + alpha * mean_value)
    return new_img

def show_image(img):
    """Print the image matrix (simulated grayscale view)."""
    for row in img:
        print(" ".join(f"{v:3}" for v in row))
    print("-" * (width * 4))

# --- Run iterations and show results ---
print("Initial Image:")
show_image(image)

for t in range(1, iterations + 1):
    image = parallel_update(image)
    print(f"Iteration {t}:")
    show_image(image)
