# cat-or-dog-keras
A deep learning model that recognizes if a given image is a photography of a dog or a photography of a cat.
Made with Keras.

## How to use it
```
python predict.py /path/to/my/image.jpg
```

## Latest results
* Training loss: 0.3206
* Validation loss: 0.3401
* Training accuracy: 0.8630
* Validation accuracy: 0.8488

## Latest training
![alt text](https://github.com/AntoineDW/cat-or-dog-keras/raw/master/training_chart.png)

```
Found 20000 images belonging to 2 classes.
Found 5000 images belonging to 2 classes.
Epoch 1/40
313/313 [==============================] - 351s 1s/step - loss: 0.6855 - acc: 0.5609 - val_loss: 0.6710 - val_acc: 0.5822
Epoch 2/40
313/313 [==============================] - 440s 1s/step - loss: 0.6514 - acc: 0.6088 - val_loss: 0.6290 - val_acc: 0.6408
Epoch 3/40
313/313 [==============================] - 337s 1s/step - loss: 0.6240 - acc: 0.6480 - val_loss: 0.6519 - val_acc: 0.6080
Epoch 4/40
313/313 [==============================] - 329s 1s/step - loss: 0.5990 - acc: 0.6767 - val_loss: 0.5789 - val_acc: 0.6978
Epoch 5/40
313/313 [==============================] - 332s 1s/step - loss: 0.5742 - acc: 0.6993 - val_loss: 0.5691 - val_acc: 0.7034
Epoch 6/40
313/313 [==============================] - 335s 1s/step - loss: 0.5574 - acc: 0.7124 - val_loss: 0.5298 - val_acc: 0.7416
Epoch 7/40
313/313 [==============================] - 331s 1s/step - loss: 0.5401 - acc: 0.7265 - val_loss: 0.5266 - val_acc: 0.7396
Epoch 8/40
313/313 [==============================] - 331s 1s/step - loss: 0.5221 - acc: 0.7406 - val_loss: 0.5148 - val_acc: 0.7442
Epoch 9/40
313/313 [==============================] - 345s 1s/step - loss: 0.5091 - acc: 0.7502 - val_loss: 0.4967 - val_acc: 0.76301
Epoch 10/40
313/313 [==============================] - 408s 1s/step - loss: 0.5005 - acc: 0.7569 - val_loss: 0.4886 - val_acc: 0.7668
Epoch 11/40
313/313 [==============================] - 380s 1s/step - loss: 0.4893 - acc: 0.7625 - val_loss: 0.4766 - val_acc: 0.7760
Epoch 12/40
313/313 [==============================] - 336s 1s/step - loss: 0.4770 - acc: 0.7735 - val_loss: 0.4790 - val_acc: 0.7692
Epoch 13/40
313/313 [==============================] - 330s 1s/step - loss: 0.4680 - acc: 0.7796 - val_loss: 0.4590 - val_acc: 0.7934
Epoch 14/40
313/313 [==============================] - 330s 1s/step - loss: 0.4599 - acc: 0.7838 - val_loss: 0.4535 - val_acc: 0.7916
Epoch 15/40
313/313 [==============================] - 336s 1s/step - loss: 0.4533 - acc: 0.7852 - val_loss: 0.4556 - val_acc: 0.7928
Epoch 16/40
313/313 [==============================] - 330s 1s/step - loss: 0.4401 - acc: 0.7944 - val_loss: 0.4465 - val_acc: 0.7990
Epoch 17/40
313/313 [==============================] - 333s 1s/step - loss: 0.4390 - acc: 0.7982 - val_loss: 0.4254 - val_acc: 0.8106
Epoch 18/40
313/313 [==============================] - 338s 1s/step - loss: 0.4277 - acc: 0.8046 - val_loss: 0.4346 - val_acc: 0.8024
Epoch 19/40
313/313 [==============================] - 416s 1s/step - loss: 0.4221 - acc: 0.8077 - val_loss: 0.4218 - val_acc: 0.8050
Epoch 20/40
313/313 [==============================] - 389s 1s/step - loss: 0.4159 - acc: 0.8084 - val_loss: 0.4111 - val_acc: 0.8128
Epoch 21/40
313/313 [==============================] - 334s 1s/step - loss: 0.4049 - acc: 0.8139 - val_loss: 0.4158 - val_acc: 0.8144
Epoch 22/40
313/313 [==============================] - 330s 1s/step - loss: 0.4009 - acc: 0.8173 - val_loss: 0.4099 - val_acc: 0.8078
Epoch 23/40
313/313 [==============================] - 329s 1s/step - loss: 0.3933 - acc: 0.8211 - val_loss: 0.4082 - val_acc: 0.8162
Epoch 24/40
313/313 [==============================] - 334s 1s/step - loss: 0.3907 - acc: 0.8224 - val_loss: 0.3985 - val_acc: 0.81903
Epoch 25/40
313/313 [==============================] - 327s 1s/step - loss: 0.3799 - acc: 0.8281 - val_loss: 0.3871 - val_acc: 0.8222
Epoch 26/40
313/313 [==============================] - 331s 1s/step - loss: 0.3761 - acc: 0.8316 - val_loss: 0.3890 - val_acc: 0.8228
Epoch 27/40
313/313 [==============================] - 333s 1s/step - loss: 0.3699 - acc: 0.8366 - val_loss: 0.3820 - val_acc: 0.8350
Epoch 28/40
313/313 [==============================] - 327s 1s/step - loss: 0.3728 - acc: 0.8332 - val_loss: 0.3785 - val_acc: 0.8314
Epoch 29/40
313/313 [==============================] - 340s 1s/step - loss: 0.3620 - acc: 0.8410 - val_loss: 0.3722 - val_acc: 0.8360
Epoch 30/40
313/313 [==============================] - 331s 1s/step - loss: 0.3606 - acc: 0.8410 - val_loss: 0.3755 - val_acc: 0.8364
Epoch 31/40
313/313 [==============================] - 336s 1s/step - loss: 0.3521 - acc: 0.8447 - val_loss: 0.3651 - val_acc: 0.8340
Epoch 32/40
313/313 [==============================] - 331s 1s/step - loss: 0.3522 - acc: 0.8444 - val_loss: 0.3623 - val_acc: 0.8412
Epoch 33/40
313/313 [==============================] - 334s 1s/step - loss: 0.3431 - acc: 0.8494 - val_loss: 0.3814 - val_acc: 0.8292
Epoch 34/40
313/313 [==============================] - 331s 1s/step - loss: 0.3430 - acc: 0.8501 - val_loss: 0.3500 - val_acc: 0.8414
Epoch 35/40
313/313 [==============================] - 331s 1s/step - loss: 0.3354 - acc: 0.8527 - val_loss: 0.3777 - val_acc: 0.8288
Epoch 36/40
313/313 [==============================] - 333s 1s/step - loss: 0.3359 - acc: 0.8518 - val_loss: 0.3567 - val_acc: 0.8460
Epoch 37/40
313/313 [==============================] - 328s 1s/step - loss: 0.3308 - acc: 0.8564 - val_loss: 0.3598 - val_acc: 0.8416
Epoch 38/40
313/313 [==============================] - 327s 1s/step - loss: 0.3245 - acc: 0.8568 - val_loss: 0.3587 - val_acc: 0.8420
Epoch 39/40
313/313 [==============================] - 332s 1s/step - loss: 0.3190 - acc: 0.8613 - val_loss: 0.3490 - val_acc: 0.8414
Epoch 40/40
313/313 [==============================] - 338s 1s/step - loss: 0.3206 - acc: 0.8630 - val_loss: 0.3401 - val_acc: 0.8488
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
