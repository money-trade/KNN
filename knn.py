import numpy as np
from collections import Counter

# จำลองตำแหน่งข้อมูล (พิกัด X, Y) ด้วย numpy array ให้สอดคล้องกับระยะห่างของเส้นประแต่ละวงในภาพ
# กำหนดให้จุด X (จุดที่ต้องการทำนาย) อยู่ที่จุดกำเนิด (0, 0)
X_target = np.array([0.0, 0.0])

# โครงสร้างข้อมูล: (หมายเลขจุด, พิกัด numpy array, 'คลาสสี')
dataset = [
    # กลุ่มที่อยู่ในวง k=1
    ('7', np.array([1.0, -1.0]), 'Red'),
    
    # กลุ่มที่อยู่ในวง k=3 (บวกจุดก่อนหน้า)
    ('4', np.array([2.0, 1.0]), 'Green'),
    ('6', np.array([2.5, 0.0]), 'Green'),
    
    # กลุ่มที่อยู่ในวง k=6 (บวกจุดก่อนหน้า)
    ('1', np.array([-3.0, 1.0]), 'Blue'),
    ('2', np.array([-2.0, 2.5]), 'Blue'),
    ('9', np.array([2.0, -3.0]), 'Red'),
    
    # กลุ่มที่อยู่นอกวง k=6 (ไม่ได้ถูกนับในโจทย์นี้ แต่ใส่ไว้ให้ครบตามภาพ)
    ('3', np.array([-1.0, 4.0]), 'Blue'),
    ('8', np.array([-3.0, -3.0]), 'Red'),
    ('5', np.array([4.0, 3.0]), 'Green')
]

def weighted_vote(k_nearest):
    """โหวตแบบถ่วงน้ำหนักด้วยระยะ: น้ำหนัก = 1 / ระยะห่าง (ยิ่งใกล้ยิ่งมีเสียงมาก)"""
    scores = {}
    for dist, name, label in k_nearest:
        scores[label] = scores.get(label, 0.0) + 1.0 / max(dist, 1e-9)
    return scores


def knn_predict(target, data, k):
    """ทำนายคลาสของ target ด้วย KNN แล้วคืนชื่อคลาส (ถ้าเสมอ ตัดสินด้วย weighted vote)"""
    print(f"\n{'='*40}")
    print(f"จำลองการทำงาน KNN ด้วย NumPy เมื่อ k = {k}")
    print(f"{'='*40}")
    
    # 1. คำนวณระยะห่าง (Euclidean Distance) ของทุกจุดพร้อมกันด้วย np.linalg.norm
    coords = np.array([c for _, c, _ in data])
    dists = np.linalg.norm(coords - target, axis=1)
    distances = [(float(d), name, label) for d, (name, _, label) in zip(dists, data)]
    
    # 2. เรียงลำดับข้อมูลจากระยะ "ใกล้" ไป "ไกล" ที่สุด
    distances.sort(key=lambda x: x[0])
    if k > len(distances):
        raise ValueError(f"k = {k} มากกว่าจำนวนข้อมูล ({len(distances)} จุด)")
    
    # 3. เลือกเพื่อนบ้านที่ใกล้ที่สุดจำนวน K ตัว
    k_nearest = distances[:k]
    
    print("เพื่อนบ้านที่พบ:")
    labels_in_k = []
    for dist, name, label in k_nearest:
        print(f"  - จุดหมายเลข {name} (สี {label}) | ระยะห่าง: {dist:.2f}")
        labels_in_k.append(label)
    
    # 4. นับผลโหวตเสียงส่วนมาก (Majority Vote)
    vote_counts = Counter(labels_in_k)
    print(f"\nผลการนับคะแนนเสียง: {dict(vote_counts)}")
    
    # 5. สรุปผลและเช็คกรณีคะแนนเสียงเสมอ (Tie)
    max_votes = max(vote_counts.values())
    winners = [label for label, count in vote_counts.items() if count == max_votes]
    
    if len(winners) > 1:
        print(f"*** นับเสียงส่วนมาก: เสมอ (Tie) ระหว่าง {', '.join(winners)} ***")
        # 6. ตัดสินด้วย Weight by distance: เพื่อนบ้านที่ใกล้กว่ามีน้ำหนักมากกว่า
        scores = weighted_vote(k_nearest)
        print("    -> ตัดสินด้วย Weight by distance (น้ำหนัก = 1 / ระยะห่าง):")
        for label in sorted(scores, key=scores.get, reverse=True):
            print(f"       {label}: {scores[label]:.3f}")
        result = max(winners, key=scores.get)
        print(f"*** สรุป: X belongs to class {result} (ตัดสินด้วยน้ำหนักระยะ) ***")
    else:
        result = winners[0]
        print(f"*** สรุป: X belongs to class {result} ***")
    return result

# รันโปรแกรมหลัก
if __name__ == "__main__":
    results = {k: knn_predict(X_target, dataset, k=k) for k in (1, 3, 6)}
    print(f"\n{'='*40}")
    print("สรุปทุกค่า k: " + ", ".join(f"k={k} -> {c}" for k, c in results.items()))