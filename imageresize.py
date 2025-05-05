from PIL import Image

img = Image.open("/workspaces/claim-flow-engine/images/Logo3.png")
resized = img.resize((800, 400))  # width, height in pixels
resized.save("/workspaces/claim-flow-engine/images/resized.png")