# Multi-Object Separation on the Pixal3D Backbone — 설계 검토

> 상태: **설계 제안 (미구현)**. Pixal3D backbone을 다객체 분리 생성으로 확장하는 방안의
> feasibility 검토와 데이터셋 조사 결과를 정리한 문서.

---

## 1. 목표

단일 이미지 + 객체별 **visible mask**(amodal 아님)를 입력으로 받아,
**N개의 객체를 개별 mesh로 분리 생성**한다.

제안 흐름:

```
Sparse Structure
  → Multi-object Sparse Structure     (segmentation mask 기반 별도 flow matching)
  → Multi-object Structured Latent    (original weights + gated additional attention)
  → SLat Decoding                     (slot별 decode → N meshes)
```

---

## 2. Backbone 실측 (설계의 전제)

코드를 읽어 확인한 사실. 일부는 초기 가정과 다르다.

| 항목 | 실제 값 | 근거 |
|---|---|---|
| SS flow model 입력 | **32³이 아니라 VAE latent `8³×8ch` (res32) / `16³×8ch` (res64)** | `configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune.json` (`resolution: 8, in_channels: 8`), `..._ft64.json` (`resolution: 16`) |
| SS 토큰 수 | 512 (res32) / 4096 (res64), dense full attention + 3D RoPE | `pixal3d/models/sparse_structure_flow.py:261` |
| SLat 토큰 수 | 8192 (256) ~ 49152 (1024), sparse | `configs/gen/slat_flow_img2shape_*` 의 `max_tokens` |
| 픽셀 정렬 | dense R³ grid를 **고정 정면 카메라**로 투영 → DINOv3 patch map에서 bilinear sampling | `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py:142` (`ProjGrid`) |
| 좌표계 | view 기준 정면 회전 → sphere normalize → **abs-max로 `[-0.5,0.5]³` 꽉 채움**, `total_scale`을 `mesh_scale`로 저장 | `data_toolkit/utils.py:496`, `data_toolkit/dual_grid_view.py:117-125` |

### 2.1 결정적으로 중요한 두 가지 성질

**(a) `ProjGrid`는 depth test를 하지 않는다.**

`project_points_to_image_batch`가 depth와 valid_mask를 반환하지만 `sample_features`는 이를 쓰지
않는다 (`image_conditioned_proj.py:218-232`). 즉 **하나의 ray 위 모든 voxel이 동일한 픽셀
feature를 받는다.** 이것이 mask 주입 방식의 성패를 좌우한다 (§5).

**(b) sparse attention이 이미 block-diagonal varlen이다.**

`q_seqlen`을 `VarLenTensor.layout`에서 뽑아 `cu_seqlens`로 flash-attn varlen을 호출한다
(`pixal3d/modules/sparse/attention/full_attn.py:100, 191`).
→ **객체를 batch entry로 펼치면 객체 내부(local) attention이 공짜로 얻어지고,
객체 간(global) attention은 같은 씬 slot들의 seqlen을 하나로 합쳐 한 번 더 호출하면 된다.**

**(c) gated weighting 템플릿이 이미 있다.**

`SparseGatedProjectAttention` (`pixal3d/modules/sparse/attention/proj_attention.py:50`)이
concat + **zero-init linear** (`:71-74`)로 "학습 초기엔 pretrained와 bit-exact 동일, 이후 점진
학습" 패턴을 구현해 두었다. 제안한 "original weights + gated additional attention"은 이 패턴을
그대로 복제하면 된다. dense 버전은 `pixal3d/modules/attention/proj_attention.py:51`.

---

## 3. 선행 연구 (설계 검증)

| 연구 | 핵심 설계 | 이 제안과의 관계 |
|---|---|---|
| **MIDI** (CVPR 2025) | pretrained image-to-3D의 self-attn 21개 중 **K=5개만** multi-instance attention으로 교체. 3D-FRONT ~15k 씬 학습, batch의 30%는 단일 객체로 정규화. **N=5**. 입력 = object RGB + **mask** + global scene image (7ch) | 거의 동일한 접근. "일부 블록만 교체" 와 "modal mask로 충분" 의 실증 |
| **PartCrafter** | 파트별 disentangled latent token + learnable identity embedding + **local/global 계층 attention**, pretrained whole-object 가중치로 초기화. 최대 16 parts / 8 objects | N-slot 토큰 설계와 ordering 문제의 선례 |
| **SceneGen** (3DV 2026) | TRELLIS backbone, **mask를 입력으로 요구**, local(에셋별) → global(씬) attention, DINOv2 + **VGGT(geometric encoder)**, position head로 상대 위치 decode | 좌표계 문제를 geometric encoder로 푼 사례 (§4) |

---

## 4. 단계별 feasibility

### Stage 1 — Multi-object Sparse Structure (난이도: 중)

- 표기 정정: `N×C×32³`이 아니라 **`N×8×16³`** (res64 기준). N=5면 20,480 tokens dense full
  attention → attention FLOPs가 단일 객체 대비 25배.
  **먼저 res32 (8³ → N×512 tokens)에서 검증할 것.** 여긴 사실상 공짜다.
- **Mask 주입은 매우 싸다.** `ProjGrid.forward`는 feature map을 인자로 받는 순수 sampler이므로
  mask map `[B,H,W,N]`을 `BHWC=True`로 그대로 넣으면 된다 (~10줄).
  게다가 mask는 full-res에서 grid_sample되므로 DINOv3 patch grid(512/16=32×32)보다 훨씬
  선명한 신호가 된다.
- `proj_in_channels`를 K만큼 넓히고 **추가 컬럼만 zero-init**하면 pretrained 동작이 정확히 보존된다.

#### ⚠️ 과소평가되기 쉬운 함정 2가지

1. **Slot permutation.** 3D RoPE는 좌표 기반이라 slot을 구분하지 못한다. learnable slot
   embedding은 물론이고 **canonical ordering 규칙이 반드시 필요**하다 (mask 면적 내림차순 등).
   flow matching은 x₁ 타겟이 결정적이어야 하므로, 같은 이미지에서 slot 순서가 흔들리면 학습이 무너진다.
2. **가변 N.** `N_max` padding + presence flag(null slot), CFG의 uncond 분기에서 slot mask 처리 규약.

### Stage 1.5 — 좌표계 (아키텍처보다 여기가 위험)

| 안 | 장점 | 단점 |
|---|---|---|
| **(A) scene frame 공유** | 위치가 자동으로 나옴, 구현 최단. **모든 slot이 `total_scale`을 공유하므로 `ProjGrid` 수정이 전혀 필요 없음** | 객체당 유효 해상도 급감. MIDI가 명시한 첫 번째 한계 |
| **(B) slot별 canonical frame + pose head** | 해상도 보존, **pretrained latent 통계 유지 → 가중치 재사용에 유리** | slot마다 다른 scale/offset으로 ProjGrid를 돌려야 함 → `assert transform_matrix is None` 2곳(`image_conditioned_proj.py:211, 264`) 해제 필요. pose를 어디선가 얻어야 함 |
| **(C) hybrid — 권장** | mask bbox + off-the-shelf metric depth(VGGT/MoGe)로 coarse layout 확보 → slot별 frustum-aligned frame → slot 내부는 pixel-aligned 생성 → layout refine | 구성요소가 하나 늘어남 |

Pixal3D의 유일한 경쟁력이 pixel alignment인데 **(A)로 가면 그 강점이 상당 부분 희석된다.**
SceneGen이 VGGT를 붙인 이유가 정확히 이것.

단, (A)를 택해도 Pixal3D는 MIDI보다 훨씬 유리하다 — MIDI는 단일 저해상도 latent였지만 Pixal3D는
512→1024→1536 cascade라 씬 대비 1/4 크기 객체도 유효 256³ 이상을 확보한다.

**결정 시점: 데이터 포맷을 좌우하므로 렌더링 시작 전에 정해야 한다.**

### Stage 2 — Multi-object SLat + gated cross-object attention (난이도: 낮음)

여기가 제일 쉽다.

- §2.1(b) 덕분에 **객체를 batch entry로 펼치면 local attention이 공짜**. cross-object attention은
  같은 씬 slot들의 seqlen을 합친 `cu_seqlens`로 attention을 한 번 더 호출 = 새 파일 수십 줄.
- Gating은 `SparseGatedProjectAttention`의 zero-init 패턴 복제 → t=0에서 pretrained와 bit-exact
  동일 → 안전한 finetune.
- **30개 블록 전부가 아니라 5~8개만** 교체 (MIDI가 K=5를 최적으로 보고). 학습 파라미터가 적어지고
  필요 데이터량이 크게 준다.

### Stage 3 — SLat Decoding (난이도: 없음)

무변경. slot별 decode → N개 mesh.

---

## 5. "visible mask만으로 충분한가" — 조건부로 성립

§2.1(a) 때문에 이 판단은 정확히 반은 맞고 반은 위험하다.

**유리한 쪽.** depth test가 없으므로 mask 값이 **ray 전체에 자동 전파**된다. "이 ray는 object i에
속한다"는 신호가 3D 전 구간에 퍼져서, 별도 back-projection 로직 없이 매우 강한 slot assignment
신호가 생긴다. 설계상 공짜로 얻는 이득.

**위험한 쪽.** object i가 j에 가려진 영역에서는 그 ray가 **j의 mask 값**을 갖는다. 따라서 slot i의
입력에 "여기엔 내가 없다"는 **틀린 음성 신호**가 들어간다. hard binary로 주면 모델이 가려진 부분을
잘라먹는 방향으로 학습될 수 있다.

**완화책**

1. slot i의 mask 채널을 **3개로**: `[mask_i, union_mask_all, mask_others]`.
   "다른 게 앞에 있어서 안 보이는 것"과 "배경이라 비어 있는 것"을 구분 가능하게.
   MIDI가 object image + mask + **global scene image**를 함께 넣은 이유.
2. occlusion ratio **커리큘럼** (0% → 70%).
3. GT는 amodal voxel이므로 loss는 그대로. 가려진 영역의 gradient를 죽이지 말 것 —
   3D prior가 amodal completion을 담당하게 하는 것이 핵심.

**결론: visible mask만으로 충분하다.** 실데이터에서 amodal mask는 얻기 매우 어렵고(합성에서만
공짜), MIDI가 modal mask + global context만으로 CVPR 2025 결과를 냈다.
단 **union mask 또는 global scene image를 반드시 함께** 넣고,
**가시율 30% 미만 객체는 평가를 분리**할 것 — 실패는 거기서 난다.

---

## 6. 데이터셋 조사

### Tier 1 — 실촬영 + 객체별 mesh + instance mask (BOP 계열)

이들의 고유 가치는 **실제 촬영 이미지 + 그에 대한 완전한 amodal 3D GT**다.
mesh와 6D pose가 둘 다 있으므로 가려진 객체의 완전한 3D 형상을 실제 사진에 대해 정확히 알 수 있다.
Objaverse 합성은 GT는 완벽하나 이미지가 가짜, 일반 실데이터는 이미지는 진짜지만 GT가 없다.
**BOP 계열만 둘 다 가진다.**

| 순위 | 데이터셋 | 규모 | 비고 |
|---|---|---|---|
| **1** | **GraspNet-1Billion** | 88 objects / 190 scenes / **97k 실촬영 RGB-D** | (프레임 × 객체) 인스턴스 수 압도적 1위. 씬당 ~10객체가 자연스럽게 쌓여 가림 통계가 실제적. tabletop 도메인이라 prior 전이 최적 |
| **2** | **YCB-Video** | 21 objects / 92 videos / ~134k frames | 학습 기여보다 **표준 비교 가능성**이 가치. FoundationPose·MegaPose 등이 전부 여기 보고 |
| **3** | **HANDAL** (BOP 버전) | **212 objects / 17 categories** / 2.2k videos / 308k frames | 실촬영 중 **객체 다양성 최대**. BOP 버전은 40개에 artist-generated 고품질 mesh + multi-object test images |
| **4** | **HOPE / HOPEv2** | 28 objects / 50 scenes | box·can 형태 → 형상 prior만으로는 못 풀고 **mask가 실제로 일해야 하는** 케이스 |
| 5 | T-LESS | 30 texture-less 대칭 산업 객체 | **진단용.** 텍스처가 없어 mask가 유일한 분리 신호 → mask conditioning 리트머스 |
| 6 | LM-O | 8 objects | 표준 occlusion 벤치마크. eval 전용 |
| 7 | BOP-Industrial (XYZ-IBD 15 parts/75 scenes/22k imgs, IPD, ITODD) | — | 극한 가림 + 대칭 + 무텍스처 bin-picking. 도메인이 멀어 학습 부적합. **최종 스트레스 테스트** |

> **FoundationPose는 데이터셋이 아니라 method다.** 학습 데이터는 Objaverse 기반 합성(LLM-보조
> 텍스처 증강)이고 평가만 YCB-V/LM-O/T-LESS/HO3D에서 한다. 참고할 것은 **합성 데이터 생성 레시피**.

**BOP가 못 주는 것 (미리 계산에 넣을 것)**

- **PBR 분해 GT 없음** (base_color/metallic/roughness) → **Stage 3(texture)는 BOP로 지도학습
  불가.** shape 단계만 finetune하고 텍스처는 Objaverse/TexVerse 학습분 유지.
- mesh 품질이 Objaverse보다 낮은 경우 많음(스캔 아티팩트, hole) → voxelize 전 정리 필요.
- **metric scale(mm)** → Pixal3D는 abs-max scale-normalized. `mesh_scale` 규약 변환 레이어 필요.
- **대칭 객체** (`models_info.json`의 `symmetries_discrete`/`continuous`)는 flow matching 타겟을
  다중모드로 만든다. §4의 slot ordering 문제와 겹쳐 증폭될 수 있다.
- 이미지가 씬 crop → Pixal3D의 rembg 기반 단일 객체 전처리 경로
  (`pixal3d/pipelines/pixal3d_image_to_3d.py:144`)를 우회하는 입력 경로 필요.

### Tier 2 — 합성 실내 씬 + 객체별 mesh

3D-FRONT / 3D-FUTURE (MIDI·SceneGen이 실제로 사용), HSSD(이미 `build_metadata.py` 지원),
Infinigen-Indoors, ProcTHOR, Scan2CAD.

### Tier 3 — 최신 compositional 합성 데이터셋 (2025~2026)

- **RecGen dataset** (arXiv 2604.27106): Objaverse-XL/ABO/HSSD에서 **198K assets**,
  **3.2M 이미지**, RGB + depth + stereo + semantic/instance mask + **amodal mask** + 6D pose +
  camera metadata. BlenderProc, 객체당 20 stereo view.
  **논문에 공개 여부 명시 없음 → 저자 문의 가치 큼.** 공개되면 아래 Phase 0을 통째로 스킵 가능.
- **Interact3D**: ~8,300 compositional scenes, 9 categories.

### Tier 4 — 직접 생성 (실질적 권장안) → §7

---

## 7. 합성 데이터 직접 생성 — 실행 가능성

### 7.1 이미 되어 있는 것 (수정 불필요)

| 기능 | 위치 | 상태 |
|---|---|---|
| **다객체 씬 정규화** | `data_toolkit/blender_script/render_cond.py:315` `normalize_scene()` | root object가 여럿이면 `ParentEmpty`로 묶어 **union bbox를 unit cube로** 정규화. 이미 다객체 대응 |
| **씬 프레이밍 자동 조정** | 같은 파일 `:367`, `:455-500` | alpha(= union mask) 기준으로 카메라 radius를 retry 조정. 다객체에서도 그대로 작동 |
| 카메라 메타 저장 | `:437-442`, `:503-509` | `aabb`, `scale`, `offset`, per-frame `transform_matrix` + `camera_angle_x` — ProjGrid 필요분 전부 |
| 분산 처리 | 모든 driver의 `--rank/--world_size` | 그대로 |

### 7.2 바꿔야 하는 것

**A. 객체별 visible mask (렌더링) — 쉬움**

`obj.pass_index = i+1` + `view_layer.use_pass_object_index = True` + compositor FileOutput로
IndexOB 저장. **IndexOB는 auxiliary pass라 추가 샘플링 비용이 사실상 0.**

amodal mask는 **학습에 불필요** — 3D GT(per-object voxel)가 이미 amodal이다. 진단용으로만
필요하면 각 객체를 단독으로 EEVEE/Workbench 재렌더(초저비용). occlusion ratio 계산에도 그대로 사용.

**B. 객체 배치 — 여기가 진짜 설계 작업**

단순 랜덤 배치는 상호 침투와 공중 부양을 만들고, 모델이 틀린 물리를 학습한다.

- support plane 위 XY footprint rejection sampling으로 **목표 가림률을 직접 제어**
  → 짧은 rigid-body settle
- 가림률은 우연에 맡기지 말고 **커리큘럼 변수로 명시 관리** (0% → 70%).
  단독 렌더 픽셀 수 대비 IndexOB 픽셀 수로 정확히 계산된다.

**C. Voxelization — `data_toolkit/dual_grid_view.py:105-125`**

- 현재: 전체 vertices concat → `sphere_normalize_torch` → `transform_mesh(view)` → `abs_max`
  → 단일 `total_scale`
- 변경: 정규화 파라미터(`sphere_radius`, `box_scale`)를 **union 기준으로 한 번만** 계산하고,
  **동일한 변환을 각 객체에 개별 적용**해 `mesh_to_flexible_dual_grid`를 객체별 호출
  → `view{XX}_obj{JJ}.vxz` N개 + `view{XX}_scale.json` 1개

> **핵심 이점**: 모든 slot이 하나의 `total_scale`을 공유하므로 **scene-frame 설계(안 A)에서
> `ProjGrid`는 수정이 전혀 필요 없다.** grid points도 mesh_scale도 동일하고, slot별로 다른 건
> 샘플링되는 채널뿐. Phase 0→2 전체가 `image_conditioned_proj.py` 무수정으로 간다.

> ⚠️ `sphere_normalize_torch`에 `assert bcenter.abs().max() < 0.25` (`data_toolkit/utils.py:549`).
> 합성 씬은 union을 먼저 센터링하지 않으면 여기서 터진다.

**D. Latent encoding** — `encode_*_latent_view.py`는 `.vxz` 단위로 돌므로 glob/naming만 변경.
다만 **인코딩 비용이 N배**.

**E. Dataset 클래스** — `pixal3d/datasets/`에 multi-object 변형 추가.
N개 latent + N개 mask 로드, `N_max` 패딩, **mask 면적 내림차순 canonical ordering**, presence flag.

### 7.3 비용 (50k 씬, N=5, 2 views)

| 단계 | 추정 | 비고 |
|---|---|---|
| 렌더링 | ~140 GPU-hours (**8 GPU면 하루**) | CYCLES 32 samples는 객체 수보다 샘플 수가 지배적. IndexOB는 공짜 |
| Voxelize | CPU, 병렬화 용이 | |
| **Latent encode** | **렌더링의 2~5배 — 병목** | 객체별로 돌아 N배. **res 256/512만 먼저, 1024는 Phase 3까지 유보** |
| 스토리지 | **실질적 제약** | 단일 객체 1024 latent도 큰데 × N × 씬. Phase 1-2는 res 256으로 시작 |

→ **엔지니어링 2~3주 + 파일럿 컴퓨트 1주.**

### 7.4 자산 큐레이션 — "Objaverse 중 high-quality mesh 기반"

repo가 이미 `min_aesthetic_score: 4.5` 필터와 `aesthetic_scores/metadata.csv` 병합
(`data_toolkit/render_cond.py:107`), `asset_stats.py`를 갖고 있다. 다객체용으로 **추가** 필요:

1. **이미 다객체인 에셋 제외.** Objaverse에는 그 자체가 씬인 에셋이 많다.
   **Objaverse++가 "multi-object composition flag"를 제공**하니 그대로 사용.
2. **Upright orientation** ← **가장 과소평가되는 비용.** Objaverse는 방향이 제멋대로인데
   support plane에 떨어뜨리려면 에셋별 canonical "up"이 필요하다. 휴리스틱(최대 평면 면적),
   gravity-aligned 서브셋, 또는 랜덤 방향 감수 중 초기에 결정할 것.
3. **상대 크기(metric scale).** Objaverse에는 실제 스케일이 없다. 머그컵과 소파가 같은 크기로
   놓이면 데이터가 망가진다. LVIS 카테고리 → 대표 치수 테이블 또는 VLM 기반 스케일 부여.

> **ABO를 먼저 쓰는 게 낫다.** 이미 `data_toolkit/datasets/ABO.py`로 배선되어 있고, 제품
> 데이터라 **실제 치수와 일관된 방향**을 가지며 mesh/PBR 품질도 Objaverse 평균보다 낫다.
> ABO로 파이프라인 검증 → Objaverse로 다양성 확장 순서를 권장.

---

## 8. 단계별 계획

| Phase | 내용 | 기간 | 리스크 |
|---|---|---|---|
| **0** | 데이터 생성: `render_cond.py` 다객체 + IndexOB, `dual_grid_view.py` 객체별 voxelize. 파일럿 5~10k 씬. **병행: RecGen 저자에게 데이터 공개 문의** | 2~3주 | 중(시간 소요형) |
| **1** ⭐ | **N=1, 가림 있는 이미지 + visible mask 채널만 추가.** `proj_in_channels` +K, 추가 컬럼 zero-init. "가려진 단일 객체의 amodal 생성"만 검증 | 1~2주 | 낮음 |
| **2** | SS N-slot (res32 → 64), slot embedding + canonical ordering | 3~4주 | 중 |
| **3** | SLat cross-object gated attention (256 → 512), 5~8 블록만 교체/학습, 30% 단일 객체 배치로 정규화 | 4~6주 | 낮음 |
| **4** | 1024 shape + texture, 실데이터 평가 (YCB-V / HOPE / GraspNet) | 3~4주 | 중 |

**Phase 1이 투자 대비 정보량이 압도적이다.** 아키텍처 변경이 거의 0(proj 채널 확장 + zero-init)인데
프로젝트 전체의 핵심 가설("visible mask만으로 amodal completion이 되는가")을 검증한다.
**여기서 안 되면 N-slot으로 가도 안 된다.**

### 평가 지표

- per-object CD / F-score (정렬 후)
- scene-level CD
- **객체 간 상호 침투(collision IoU) 비율** ← cross-object attention이 실제로 일하는지 보는 유일한 척도
- **가시율 구간별 분해 성능** (특히 <30%)

---

## 9. 종합 판정

**Feasibility: 높음.** 선행 연구 3건이 핵심 설계를 검증했고, 이 repo는 varlen block-diagonal
attention과 zero-init gated fusion을 이미 갖추고 있어 시작점이 좋다.

난이도 순위:
`데이터 생성 > 좌표계/pose 설계 > SS N-slot > SLat cross-object attention > decoding(무변경)`

**주된 리스크는 아키텍처가 아니라 데이터와 좌표계 결정이다.**

### 착수 전 결정 필요

1. **자산 풀**: ABO 우선 vs Objaverse
2. **좌표계**: scene frame (A) vs hybrid (C) — **데이터 포맷을 좌우하므로 렌더링 전에 확정**

---

## 참고 문헌

- [BOP: Benchmark for 6D Object Pose Estimation — Datasets](https://bop.felk.cvut.cz/datasets/)
- [BOP Challenge 2025](https://bop.felk.cvut.cz/challenges/)
- [BOP Challenge 2024 on Model-Based and Model-Free 6D Object Pose Estimation](https://arxiv.org/abs/2504.02812)
- [MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation (CVPR 2025)](https://arxiv.org/abs/2412.03558)
- [SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass (3DV 2026)](https://arxiv.org/abs/2508.15769)
- [PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers](https://arxiv.org/abs/2506.05573)
- [Reconstruction by Generation (RecGen): 3D Multi-Object Scene Reconstruction from Sparse Observations](https://arxiv.org/abs/2604.27106)
- [Interact3D: Compositional 3D Generation of Interactive Objects](https://arxiv.org/abs/2603.16085)
- [HANDAL: A Dataset of Real-World Manipulable Object Categories](https://arxiv.org/abs/2308.01477)
- [XYZ-IBD: A High-precision Bin-picking Dataset for Object 6D Pose Estimation](https://arxiv.org/abs/2506.00599)
- [HOT3D: An Egocentric Dataset for 3D Hand and Object Tracking](https://arxiv.org/abs/2406.09598)
- [BlenderProc — BOP Scene Replication](https://dlr-rm.github.io/BlenderProc/examples/datasets/bop_scene_replication/README.html)
