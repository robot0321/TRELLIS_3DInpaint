# TRELLIS 3D Inpaint

## TOODs
### System
1. [v] Implement save/load Gaussian outputs (example_inpaint.py)
2. [v] Debugging optimize_initnoise_sparse_structure(), 
3. [v] Debugging optimize_initnoise_slat()
4. [ ] 기본 기능 작동 확인 후에 commit 


### Method
1. gradient masking -> optimize 안정성 (됬나? 코드 체크 필요)
2. feature alignment (N(0,1) 혹은...)
3. Apply FDG to 3D
4. spatial 말고 안정성 있는 구조 필요 
    - wavelet이나 Non-uniform FFT
    - geometry는 안정성 덕에 높은 lr이 가능하고 slat은 아니라서 낮은 lr 써야하는건가?

### Experiment
1. Implement evaluation code - eval while optimize initnoise & final eval
    - [v] geometry on iou, dice 
    - [v] color on Mean color error - 수정 필요
    - [ ] evaluation 비교하기 편한 방식으로 기록
2. Implement Baselien code (RePaint, MultiDiffusion)