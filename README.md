# TRELLIS 3D Optinit

* TRELLIS 기본 설치
* 기본 돌리는 코드는 `example_optinit.py` 보면 됌
* toys4k 돌리는 코드는 example_toys4k.py인데, 비교 방법들이랑 전부 돌리는 코드라 일단은 example_optinit.py에서 작업하면 될 듯
* toys4k 데이터는 selene:/home/chung/workspace/TRELLIS_3DInpaint/datasets에 있는데, nfs에서 당겨서 쓰는거라 사용하는 서버에서 nfs 마운트 해서 쓰면 될 듯
* 아래 내용은 무시 

---

## TOODs
### System
1. [v] Implement save/load Gaussian outputs (example_optinit.py)
2. [v] Debugging optimize_initnoise_sparse_structure(), 
3. [v] Debugging optimize_initnoise_slat()
4. [v] 기본 기능 작동 확인 후에 commit 


### Method
1. [ ] gradient masking -> optimize 안정성 (됬나? 코드 체크 필요)
2. [ ] feature alignment (N(0,1) 혹은...) -> 값을 찾아야 되나? 
3. [ ] Apply FDG to 3D
4. [ ] spatial 말고 안정성 있는 구조 필요 
    - wavelet이나 Non-uniform FFT
    - geometry는 안정성 덕에 높은 lr이 가능하고 slat은 아니라서 낮은 lr 써야하는건가?
5. [ ] new priorloss = mean(0), var(1), skew(2), kurtosis(3) + labmda

### Experiment
1. Implement evaluation code - eval while optimize initnoise & final eval
    - [v] geometry on iou, dice 
    - [v] color on Mean color error - 수정 필요
    - [ ] evaluation 비교하기 편한 방식으로 기록
2. Implement Baselien code (RePaint, MultiDiffusion)
