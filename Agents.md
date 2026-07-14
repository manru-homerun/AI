## ? 1. 브랜치 생성 전략

`main + 작업 브랜치` 구조를 사용한다. `main`에는 직접 push하지 않으며, 모든 변경은 작업 브랜치에서 진행한 후 Pull Request로 병합한다.

### 브랜치 이름

```
<type>/<issue-number>-<description>
```

| Type       | 용도                  | 예시                           |
| ---------- | --------------------- | ------------------------------ |
| `feat`     | 기능 추가             | `feat/42-gru-decoder`          |
| `fix`      | 버그 수정             | `fix/43-padding-mask`          |
| `data`     | 데이터 및 전처리 변경 | `data/44-merge-region-csv`     |
| `exp`      | 모델 실험             | `exp/45-hidden-size`           |
| `refactor` | 구조 개선             | `refactor/46-split-preprocess` |
| `test`     | 테스트 변경           | `test/47-schema-validation`    |
| `docs`     | 문서 변경             | `docs/48-training-guide`       |
| `chore`    | 환경 및 설정 변경     | `chore/49-add-ci`              |

### 생성 방법

```bash
git switch main
git pull --ff-only
git switch -c feat/42-gru-decoder
```

### 운영 규칙

- 하나의 브랜치는 하나의 Issue만 처리한다.
- 브랜치는 가능한 한 짧게 유지한다.
- 병합된 브랜치는 삭제한다.
- 공유 브랜치와 `main`에는 force push하지 않는다.
- 개인 브랜치의 rebase 이후에만 `git push --force-with-lease`를 허용한다.

---

## ?2. 커밋 전략

커밋은 하나의 논리적 변경 단위로 작성한다. 기능 추가, 리팩터링, 코드 포맷 변경은 가능한 한 별도 커밋으로 분리한다.

### 커밋 메시지

```
<type>(<scope>): <summary> (<issue-number>)
```

예시:

```
feat(model): add GRU travel sequence decoder
fix(data): preserve visit order after merge
refactor(preprocess): extract regional file loader
test(model): add padding mask test
chore(ci): run pytest on pull requests
```

### 커밋 타입

| Type       | 의미                     |
| ---------- | ------------------------ |
| `feat`     | 기능 추가                |
| `fix`      | 오류 수정                |
| `data`     | 데이터 및 전처리 변경    |
| `exp`      | 실험 코드 및 설정 변경   |
| `refactor` | 기능 변경 없는 구조 개선 |
| `test`     | 테스트 추가 및 수정      |
| `docs`     | 문서 변경                |
| `chore`    | 환경, 의존성, CI 설정    |
| `perf`     | 성능 개선                |

### Scope 규칙

```jsx
data;
schema;
preprocess;
dataset;
model;
train;
eval;
infer;
serving;
config;
ci;
docs;
```

### 작성 규칙

- 제목은 영문 명령형으로 작성한다.
- `update`, `수정`, `작업 완료`처럼 모호한 표현을 사용하지 않는다.
- 커밋 하나만 되돌려도 변경 의미가 유지되어야 한다.
- 모델 가중치, 원본 데이터, API Key는 커밋하지 않는다.
- 커밋 전 변경 내용을 확인한다.

```bash
git status
git diff
git add <변경한 파일>
git diff --staged
git commit -m "feat(model): add GRU decoder"
```
