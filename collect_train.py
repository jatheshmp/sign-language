"""Optional: train your own alphabet/word classifier.
  python collect_train.py collect   -> press a letter key to save a sample of that sign, ESC to quit
  python collect_train.py train     -> creates model.pkl (app.py picks it up automatically)
Collect ~100 samples per sign, varying hand angle and distance.
"""
import sys, csv, os, pickle
import cv2, numpy as np, mediapipe as mp

mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils


def features(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm])
    pts -= pts[0]
    return (pts / (np.abs(pts).max() + 1e-6)).flatten()


def collect():
    cap = cv2.VideoCapture(0)
    n = 0
    with mp_hands.Hands(max_num_hands=1) as hands, open("data.csv", "a", newline="") as f:
        w = csv.writer(f)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None
            if lm:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"saved: {n} | press key = label, ESC = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("collect", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if lm and 32 < k < 127:
                w.writerow([chr(k).upper(), *features(lm.landmark)])
                n += 1
    cap.release()
    cv2.destroyAllWindows()


def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    rows = list(csv.reader(open("data.csv")))
    y = [r[0] for r in rows]
    X = np.array([[float(v) for v in r[1:]] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)
    clf = RandomForestClassifier(n_estimators=200).fit(Xtr, ytr)
    print("Test accuracy:", clf.score(Xte, yte))
    pickle.dump(clf, open("model.pkl", "wb"))


if __name__ == "__main__":
    collect() if sys.argv[1:] == ["collect"] else train()
