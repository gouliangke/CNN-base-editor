from model import *
from load_data_test import *
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def write_pred(y_true, y_pred):
	with open('log.txt', 'w') as fo:
		auroc = metrics.roc_auc_score(y_true, y_pred)
		aupr = metrics.average_precision_score(y_true, y_pred)
		fo.write("# auroc=%.4f\n"%auroc)
		fo.write("# aupr=%.4f\n"%aupr)
		for i in range(len(y_true)):
			fo.write("{}\t{}\n".format(y_true[i], y_pred[i]))


X, Y = load_data()
X_testpi = test_data()
model = build_model()

checkpointer = ModelCheckpoint(filepath="bestmodel.h5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


X_train, tmp_X, Y_train, tmp_y = train_test_split(
	X, Y, test_size = 0.2, random_state=111)
X_val, X_test, Y_val, Y_test = train_test_split(
	tmp_X, tmp_y, test_size=0.5, random_state=222)

history = model.fit(X_train, Y_train,
	validation_data = [X_val, Y_val],
	shuffle=True,
	epochs=50, 
	batch_size=64,
	callbacks=[checkpointer, earlystopper],
	verbose=1)


print(model.evaluate(X_val, Y_val))
print(model.evaluate(X_test, Y_test))
y_test_pred = model.predict(X_test).flatten()
write_pred(Y_test, y_test_pred)

y_test_pred2 = model.predict(X_testpi).flatten()
print(y_test_pred2)

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('acc.png')
plt.gcf().clear()

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('loss.png')
plt.gcf().clear()
