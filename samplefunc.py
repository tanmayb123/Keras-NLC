import data_helpers
import keras
import sys
from keras.utils import plot_model

# Gather input and filter first 269 characters from data
input = sys.argv[1][0:270]

# Pre-process the input through data_helpers
x = data_helpers.filterinput(input)

# Load pre-trained model
model = keras.models.load_model('model_review.h5')

# Run pre-processed input through the model and gather the prediction
y = model.predict(x)

# Print model output
resultstr = ""

if round(y) == 1.0:
        resultstr += "Not Spam"
else:
        resultstr += "Spam"

print resultstr
