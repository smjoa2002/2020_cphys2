{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled1.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "mount_file_id": "1kCQExb-_tUQNMufDi2R55EIXeWKNsU6L",
      "authorship_tag": "ABX9TyPhX3mL2VU77Wt8YSi4Lhj1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/smjoa2002/2020_cphys2/blob/master/hw7-1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T0wQkX0IwXGh"
      },
      "source": [
        "from keras.datasets import mnist\n",
        "from tensorflow import keras\n",
        "from keras import models\n",
        "from keras import layers\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from keras.utils import to_categorical\n",
        "\n",
        "(X,y), (Xtest,y_test)=mnist.load_data()\n",
        "y_train=to_categorical(y)\n",
        "y_test=to_categorical(y_test)\n",
        "\n",
        "net=models.Sequential()\n",
        "net.add(layers.Dense(10, activation='softmax', input_shape=(28*28,)))\n",
        "net.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])\n",
        "\n",
        "\n"
      ],
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aSsXF96s1oBE"
      },
      "source": [
        "X_train=X.reshape((60000,28*28))\n",
        "X_train=X_train/255\n",
        "\n",
        "X_test=Xtest.reshape((10000,28*28))\n",
        "X_test=X_test/255\n"
      ],
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TBfAWtXl2pVw",
        "outputId": "9c0020be-e853-48f7-82d5-718b58215519",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 746
        }
      },
      "source": [
        "net.fit(X_train,y_train, epochs=20)"
      ],
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/20\n",
            "1875/1875 [==============================] - 1s 597us/step - loss: 0.7697 - accuracy: 0.8194\n",
            "Epoch 2/20\n",
            "1875/1875 [==============================] - 1s 611us/step - loss: 0.4561 - accuracy: 0.8806\n",
            "Epoch 3/20\n",
            "1875/1875 [==============================] - 1s 600us/step - loss: 0.4034 - accuracy: 0.8911\n",
            "Epoch 4/20\n",
            "1875/1875 [==============================] - 1s 607us/step - loss: 0.3770 - accuracy: 0.8965\n",
            "Epoch 5/20\n",
            "1875/1875 [==============================] - 1s 593us/step - loss: 0.3602 - accuracy: 0.9005\n",
            "Epoch 6/20\n",
            "1875/1875 [==============================] - 1s 593us/step - loss: 0.3482 - accuracy: 0.9034\n",
            "Epoch 7/20\n",
            "1875/1875 [==============================] - 1s 617us/step - loss: 0.3392 - accuracy: 0.9054\n",
            "Epoch 8/20\n",
            "1875/1875 [==============================] - 1s 599us/step - loss: 0.3321 - accuracy: 0.9078\n",
            "Epoch 9/20\n",
            "1875/1875 [==============================] - 1s 610us/step - loss: 0.3263 - accuracy: 0.9096\n",
            "Epoch 10/20\n",
            "1875/1875 [==============================] - 1s 597us/step - loss: 0.3214 - accuracy: 0.9100\n",
            "Epoch 11/20\n",
            "1875/1875 [==============================] - 1s 606us/step - loss: 0.3172 - accuracy: 0.9115\n",
            "Epoch 12/20\n",
            "1875/1875 [==============================] - 1s 626us/step - loss: 0.3136 - accuracy: 0.9126\n",
            "Epoch 13/20\n",
            "1875/1875 [==============================] - 1s 598us/step - loss: 0.3103 - accuracy: 0.9141\n",
            "Epoch 14/20\n",
            "1875/1875 [==============================] - 1s 593us/step - loss: 0.3073 - accuracy: 0.9144\n",
            "Epoch 15/20\n",
            "1875/1875 [==============================] - 1s 604us/step - loss: 0.3049 - accuracy: 0.9153\n",
            "Epoch 16/20\n",
            "1875/1875 [==============================] - 1s 600us/step - loss: 0.3025 - accuracy: 0.9160\n",
            "Epoch 17/20\n",
            "1875/1875 [==============================] - 1s 595us/step - loss: 0.3003 - accuracy: 0.9166\n",
            "Epoch 18/20\n",
            "1875/1875 [==============================] - 1s 587us/step - loss: 0.2985 - accuracy: 0.9171\n",
            "Epoch 19/20\n",
            "1875/1875 [==============================] - 1s 589us/step - loss: 0.2967 - accuracy: 0.9174\n",
            "Epoch 20/20\n",
            "1875/1875 [==============================] - 1s 604us/step - loss: 0.2950 - accuracy: 0.9179\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tensorflow.python.keras.callbacks.History at 0x7fa6557a5780>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "B7o-kisj3gFr",
        "outputId": "e24fd0f3-5ebf-423d-ffd9-32949b392382",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 283
        }
      },
      "source": [
        "plt.imshow(Xtest[9])"
      ],
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7fa653629208>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mBlcfEVB3rQL",
        "outputId": "af027c36-9211-451d-cfcf-ab7d8dc7d0a5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        }
      },
      "source": [
        "np.argmax(net.predict(X_test)[9])"
      ],
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "9"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ouw5SBYk4Kdj",
        "outputId": "6afdc736-bd0b-4e35-f619-b55ac75bb08a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        }
      },
      "source": [
        "y_test[8]"
      ],
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=float32)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IeW6-Nkh4Zav",
        "outputId": "2c8f3b94-677e-4bbf-a2b3-3b17cbf5e5c8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        }
      },
      "source": [
        "net.evaluate(X_test,y_test)"
      ],
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "313/313 [==============================] - 0s 601us/step - loss: 0.2869 - accuracy: 0.9201\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[0.2868898808956146, 0.9200999736785889]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 25
        }
      ]
    }
  ]
}