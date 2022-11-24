clear
clc

% --- MAP METHOD ---

% load the data
load("Alpha.mat");  % 1 x 9 vector
load("Prior_2.mat");    % Strategy 1 (W0 mu0_FG mu0_BG)
load("TrainingSamplesDCT_subsets_8.mat", "D1_BG", "D1_FG");   % D1: cheetah & grass
load("ZigZagVec.mat")   % 1 x 64 vector
load("cheetahMat.txt")  % our image
load("cheetah_mask.mat")    % our ideal mask

% get the size of the image
[m, n] = size(cheetahMat);
m = m - 7;
n = n - 7;
maskRes = zeros(m, n);
totalPixels = m * n;

% # of samples
[n_cheetah, ~] = size(D1_FG);
[n_grass, ~]   = size(D1_BG);
totalSamples = (n_cheetah + n_grass);

% calculate the priors
Prior_cheetah = n_cheetah / totalSamples;
Prior_grass   = n_grass   / totalSamples;

% mu_MLs and covs
mu_ML_cheetah = mean(D1_FG);
mu_ML_grass   = mean(D1_BG);
cov_cheetah = cov(D1_FG);
cov_grass   = cov(D1_BG);

% calculate the sigma_0
d = 6;  % which alpha am I using
sigma_0 = diag(alpha(d) .* W0);
    
% calculate the mu_ns
mu_n_cheetah = ((n_cheetah .* sigma_0) ./ (cov_cheetah + n_cheetah .* sigma_0)) .* mu_ML_cheetah + ((cov_cheetah) ./ (cov_cheetah + n_cheetah .* sigma_0)) .* mu0_FG;
mu_n_grass   = ((n_grass   .* sigma_0) ./ (cov_grass   + n_grass   .* sigma_0)) .* mu_ML_grass   + ((cov_grass)   ./ (cov_grass   + n_grass   .* sigma_0)) .* mu0_BG;

% calculate SIGMAs (sigma) and their inverses
SIGMA_cheetah = cov_cheetah;
SIGMA_grass   = cov_grass;
SIGMAInv_cheetah = inv(SIGMA_cheetah);
SIGMAInv_grass   = inv(SIGMA_grass);

% calculate the coefficients
deno_cheetah = (sqrt(((2 * pi)^64) * det(SIGMA_cheetah)));
deno_grass   = (sqrt(((2 * pi)^64) * det(SIGMA_grass)));

% calculate DCT2 (64D)
error = 0;
for i = 1 : m
    for j = 1 : n
        Block = cheetahMat(i : i + 7, j : j + 7);
        Block_DCT = dct2(Block, 8, 8);
        V = Block_DCT(:).';
        X = zeros(1, 64);

        % mapping
        for k = 1 : 64
            X(ZigZagVec(k)) = V(k);
        end

        % calculate the probabilities
        % G(X, mu_n, SIGMA)
        P_X_likelihood_cheetah = exp(-0.5 * (X - mu_n_cheetah) * (SIGMAInv_cheetah) * (X - mu_n_cheetah).') / deno_cheetah;
        P_X_likelihood_grass   = exp(-0.5 * (X - mu_n_grass) * (SIGMAInv_grass) * (X - mu_n_grass).') / deno_grass;
        P_X_cheetah = P_X_likelihood_cheetah * Prior_cheetah;
        P_X_grass   = P_X_likelihood_grass   * Prior_grass;

        % based on the results, decide whether the pixel is grass or cheetah
        if (P_X_grass > P_X_cheetah)
            maskRes(i, j) = 0;
        else
            maskRes(i, j) = 1;
        end

        % calculate the error
        if (P_X_grass > P_X_cheetah)
            result = 0;
        else
            result = 1;
        end

        % calculate the error rate
        if(result ~= cheetah_mask(i, j))
            error = error + 1;
        end

    end
end

% display my mask
imshow(maskRes)
title("MAP Method")

% result (error)
errorRate = error / totalPixels;
disp(errorRate)