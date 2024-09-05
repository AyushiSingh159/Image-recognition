{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c99aa01b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:3000\n",
      "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:01:54] \"GET / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:01:54] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 886ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:02:04] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 206ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:02:15] \"POST / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:09:27] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 205ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:09:35] \"POST / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:12:22] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 224ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:12:29] \"POST / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:13:19] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 214ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:13:28] \"POST / HTTP/1.1\" 200 -\n",
      "INFO:werkzeug:127.0.0.1 - - [29/Jun/2024 16:14:43] \"GET / HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request, send_from_directory\n",
    "from keras.preprocessing.image import load_img, img_to_array\n",
    "from tensorflow.keras.models import load_model\n",
    "import numpy as np\n",
    "import os\n",
    "\n",
    "# Define the absolute path to your templates directory\n",
    "TEMPLATES_DIR = r\"C:\\Users\\KIIT\\Downloads\\Flower_recognition\\temp\"\n",
    "\n",
    "app = Flask(__name__, template_folder=TEMPLATES_DIR)\n",
    "\n",
    "# Load your custom CNN model\n",
    "model = load_model('cnn_model.h5')\n",
    "\n",
    "@app.route('/', methods=['GET'])\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/', methods=['POST'])\n",
    "def predict():\n",
    "    if 'imagefile' not in request.files:\n",
    "        return \"No file part\"\n",
    "    \n",
    "    imagefile = request.files['imagefile']\n",
    "    \n",
    "    if imagefile.filename == '':\n",
    "        return \"No selected file\"\n",
    "    \n",
    "    image_path = os.path.join(\"./images\", imagefile.filename)\n",
    "    \n",
    "    if not os.path.exists('./images'):\n",
    "        os.makedirs('./images')\n",
    "    \n",
    "    imagefile.save(image_path)\n",
    "\n",
    "    image = load_img(image_path, target_size=(64, 64))  # Use the same target size as your trained model\n",
    "    image = img_to_array(image)\n",
    "    image = np.expand_dims(image, axis=0)\n",
    "    image = image / 255.0  # Rescale the image to match your training data preprocessing\n",
    "\n",
    "    yhat = model.predict(image)\n",
    "    class_indices = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}\n",
    "    prediction = class_indices[np.argmax(yhat)]\n",
    "\n",
    "    classification = '%s (%.2f%%)' % (prediction, np.max(yhat) * 100)\n",
    "\n",
    "    return render_template('index.html', prediction=classification)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(port=3000, debug=True, use_reloader=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e149a70d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
