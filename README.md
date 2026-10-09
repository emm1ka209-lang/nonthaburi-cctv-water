# nonthaburi-cctv-water
ระบบกล้องวงจรปิดตรวจวัดระดับน้ำ จังหวัดนนทบุรี (FastAPI + หน้าเว็บ static)

## อัปเดตระดับน้ำ
ตั้ง `ADMIN_TOKEN` ใน Render แล้วเรียก:

    curl -X POST https://<โดเมน>/api/water/CCTV-01 \
      -H "Authorization: Bearer $ADMIN_TOKEN" -H "Content-Type: application/json" \
      -d '{"value": 42.5, "status": "warning"}'

`status` = normal | warning | critical (ข้อความธงสร้างให้อัตโนมัติ)
หมายเหตุ: Render free plan ล้างไฟล์เมื่อ deploy ใหม่ ค่าใน `data/water.json` จะกลับเป็นค่าที่ commit ไว้

## Environment variables
ADMIN_TOKEN, ALLOWED_ORIGINS, SNAPSHOT_TTL (วินาที, ค่าเริ่มต้น 10), DATA_DIR
