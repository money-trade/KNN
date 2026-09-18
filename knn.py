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

def knn_predict(target, data, k):
    print(f"\n{'='*40}")
    print(f"จำลองการทำงาน KNN ด้วย NumPy เมื่อ k = {k}")
    print(f"{'='*40}")
    
    # 1. คำนวณระยะห่าง (Euclidean Distance) ด้วย np.linalg.norm
    distances = []
    for name, coords, label in data:
        dist = np.linalg.norm(coords - target)
        distances.append((dist, name, label))
    
    # 2. เรียงลำดับข้อมูลจากระยะ "ใกล้" ไป "ไกล" ที่สุด
    distances.sort(key=lambda x: x[0])
    
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
        print(f"*** สรุป: X belongs to class เสมอ (Tie) ระหว่าง {', '.join(winners)} ***")
        print("    -> ไม่สามารถฟันธงได้ด้วยวิธีนับเสียงส่วนมากปกติ (ต้องใช้วิธีลดค่า k หรือ Weight by distance)")
    else:
        print(f"*** สรุป: X belongs to class {winners[0]} ***")

# รันโปรแกรมหลัก
if __name__ == "__main__":
    knn_predict(X_target, dataset, k=1)
    knn_predict(X_target, dataset, k=3)
    knn_predict(X_target, dataset, k=6)