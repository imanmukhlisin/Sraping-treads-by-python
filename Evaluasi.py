from sklearn.metrics import classification_report, accuracy_score

class Evaluator:
    def evaluate(self, y_true, y_pred):
        print("Akurasi:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))