clear
clc

load("HW2\TrainingSamplesDCT_8_new.mat");
img_cheetah = imread("cheetah.bmp");
mask_ans = imread("cheetah_mask.bmp");
img_cheetah = im2double(img_cheetah);
mask_ans = imbinarize(mask_ans);

subplot(1, 3, 1);
imshow(mask_ans);
hold on
title('Ideal Mask')

% creat vectors for means and variances
mean_cheetah_64D = zeros(1, 64);
var_cheetah_64D  = zeros(1, 64);
mean_grass_64D   = zeros(1, 64);
var_grass_64D    = zeros(1, 64);  

% calculate means and variances of cheetah and grass
for i = 1 : 64
    col = TrainsampleDCT_FG(:,i);
    mean_cheetah_64D(1, i) = mean(col);
    var_cheetah_64D(1, i) = var(col);
    col = TrainsampleDCT_BG(:,i);
    mean_grass_64D(1, i) = mean(col);
    var_grass_64D(1, i) = var(col);
end

% DCT (64D)
[m, n] = size(img_cheetah);
mask_64D = zeros(m - 7, n - 7);

% padding the image
I = zeros(m + 7, n + 7);
I(1:m, 1:n) = img_cheetah;

% the zigzag map
ZigZagM = [0   1   5   6  14  15  27  28; 
    2   4   7  13  16  26  29  42;
    3   8  12  17  25  30  41  43; 
    9  11  18  24  31  40  44  53;
    10  19  23  32  39  45  52  54;
    20  22  33  38  46  51  55  60;
    21  34  37  47  50  56  59  61;
    35  36  48  49  57  58  62  63];
ZigZagM = ZigZagM + 1;  % index in MATLAB starts with 1
ZigZagV = ZigZagM(:);   % 64 x 1
ZigZagV = ZigZagV.';    % 1 x 64

% calculate the sigmas and sigma^-1s
sigma_cheetah_64D    = cov(TrainsampleDCT_FG);
sigma_grass_64D      = cov(TrainsampleDCT_BG);
sigmaInv_cheetah_64D = inv(sigma_cheetah_64D);
sigmaInv_grass_64D   = inv(sigma_grass_64D);

% calculate the coefficients
deno_cheetah = (sqrt(((2 * pi)^64) * det(sigma_cheetah_64D)));
deno_grass   = (sqrt(((2 * pi)^64) * det(sigma_grass_64D)));

% calculate DCT2 (64D)
m = m - 7;
n = n - 7;
for i = 1 : m
    for j = 1 : n
        Block = I(i : i + 7, j : j + 7);
        Block_DCT = dct2(Block, 8, 8);
        V = Block_DCT(:).';
        X = zeros(1, 64);
        % mapping
        for k = 1 : 64
            X(ZigZagV(k)) = V(k);
        end
        % calculate the probabilities
        P_X_cheetah = exp(-0.5 * (X - mean_cheetah_64D) * (sigmaInv_cheetah_64D) * (X - mean_cheetah_64D).') / deno_cheetah;
        P_X_grass   = exp(-0.5 * (X - mean_grass_64D)   * (sigmaInv_grass_64D)   * (X - mean_grass_64D).')   / deno_grass;
        % based on the results, decide whether the pixel is grass or cheetah
        if (P_X_grass > P_X_cheetah)
            mask_64D(i, j) = 0;
        else
            mask_64D(i, j) = 1;
        end
    end
end

% compare my mask with ans
e_64D = 0;
for i = 1 : m
    for j = 1 : n
        if (mask_ans(i, j) ~= mask_64D(i, j))
            e_64D = e_64D + 1;
        end
    end
end
error_rate_64D = e_64D / (m * n);

% plot the result
subplot(1, 3, 2);
imshow(mask_64D);
hold on
title('Mask (64 Features)')

% creat vectors for means and variances (8D)
mean_cheetah_8D = zeros(1, 8);
var_cheetah_8D  = zeros(1, 8);
mean_grass_8D   = zeros(1, 8);
var_grass_8D    = zeros(1, 8);  

% calculate means and variances of cheetah and grass
for i = 1 : 8
    col = TrainsampleDCT_FG(:,i);
    mean_cheetah__8D(1, i) = mean(col);
    var_cheetah__8D(1, i) = var(col);
    col = TrainsampleDCT_BG(:,i);
    mean_grass_8D(1, i) = mean(col);
    var_grass_8D(1, i) = var(col);
end

% DCT (64D)
[m, n] = size(img_cheetah);
mask_8D = zeros(m - 7, n - 7);

% padding the image
I = zeros(m + 7, n + 7);
I(1:m, 1:n) = img_cheetah;

% calculate the sigmas and sigma^-1s
sigma_cheetah_8D    = cov(TrainsampleDCT_FG(:, 8));
sigma_grass_8D      = cov(TrainsampleDCT_BG(:, 8));
sigmaInv_cheetah_8D = inv(sigma_cheetah_8D);
sigmaInv_grass_8D   = inv(sigma_grass_8D);

% calculate the coefficients
deno_cheetah_8D = (sqrt(((2 * pi)^8) * det(sigma_cheetah_8D)));
deno_grass_8D   = (sqrt(((2 * pi)^8) * det(sigma_grass_8D)));

% calculate DCT2 (64D)
m = m - 7;
n = n - 7;
for i = 1 : m
    for j = 1 : n
        Block = I(i : i + 7, j : j + 7);
        Block_DCT = dct2(Block, 8, 8);
        V = Block_DCT(:).';
        X = zeros(1, 64);
        % mapping
        for k = 1 : 64
            X(ZigZagV(k)) = V(k);
        end
        % calculate the probabilities
        P_X_cheetah = exp(-0.5 * (X(1:8) - mean_cheetah_8D) * (sigmaInv_cheetah_8D) * (X(1:8) - mean_cheetah_8D).') / deno_cheetah_8D;
        P_X_grass   = exp(-0.5 * (X(1:8) - mean_grass_8D)   * (sigmaInv_grass_8D)   * (X(1:8) - mean_grass_8D).')   / deno_grass_8D;
        % based on the results, decide whether the pixel is grass or cheetah
        if (P_X_grass > P_X_cheetah)
            mask_8D(i, j) = 0;
        else
            mask_8D(i, j) = 1;
        end
    end
end

% compare my mask with ans
e_8D = 0;
for i = 1 : m
    for j = 1 : n
        if (mask_ans(i, j) ~= mask_8D(i, j))
            e_8D = e_8D + 1;
        end
    end
end
error_rate_8D = e_8D / (m * n);

% plot the result
subplot(1, 3, 3);
imshow(mask_8D);
hold on
title('Mask (8 Features)')
figure

imshow(mask_64D);
title('Mask (64 Features)')
figure
imshow(mask_8D);
title('Mask (8 Features)')