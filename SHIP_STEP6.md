# STEP 6: Push and Post (직접 실행)

현재 저장소는 원본 `666ghj/MiroFish` 클론본입니다.  
**본인 GitHub에서 Fork한 뒤** 아래 중 하나로 진행하세요.

## A) Fork 후 이 폴더를 본인 Fork와 연결

```powershell
cd C:\Users\baesy\MiroFish
git remote rename origin upstream
git remote add origin https://github.com/BAESY2/MiroFish.git
git add .
git commit -m "feat: add Temporal Inversion Engine (Bayesian DAG reversal)"
git push -u origin main
```

## B) Fork한 저장소를 새로 클론한 뒤 변경사항만 복사

- `backend/inversion/` ← 여기 있는 전체 폴더
- `backend/test_inversion/` ← 여기 있는 전체 폴더
- `backend/app/api/inversion_routes.py`
- `backend/app/__init__.py` (inversion_bp 등록 부분)
- `backend/app/api/__init__.py` (inversion_bp import)
- `backend/requirements.txt` (networkx 추가)
- `README.md` (Temporal Inversion 섹션)

---

## 포스팅 문구 (복사해서 사용)

**X/Twitter**
> Forked MiroFish and added Tenet-style temporal inversion. It reverses Bayes' theorem on the prediction graph to answer "what HAD to be true?" Pure math, 0 LLM calls, 46ms. github.com/BAESY2/MiroFish

**Reddit r/MachineLearning**  
동일 메시지 + 링크

**Hacker News**
> Show HN: Temporal Inversion for MiroFish — Bayesian DAG reversal finds what had to be true for predictions to hold
