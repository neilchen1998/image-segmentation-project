clear
clc

% --- ML METHOD ---

% load the data
load("Alpha.mat");  % 1 x 9 vector
load("TrainingSamplesDCT_subsets_8.mat", "D4_BG", "D4_FG");   % D1: cheetah & grass
load("ZigZagVec.mat")   % 1 x 64 vector
load("cheetahMat.txt")  % our image
load("cheetah_mask.mat")    % our ideal mask

% get the size of the image
[m, n] = size(cheetahMat);
m = m - 7;
n = n - 7;
totalPixels = m * n;
maskRes = zeros(m, n);

% # of samples
[n_cheetah, ~] = size(D4_FG);
[n_grass, ~]   = size(D4_BG);
totalSamples = (n_cheetah + n_grass);

% calculate the priors
Prior_cheetah = n_cheetah / totalSamples;
Prior_grass   = n_grass   / totalSamples;

% mus and covs
mu_cheetah = mean(D4_FG);
mu_grass   = mean(D4_BG);
cov_cheetah = cov(D4_FG);
cov_grass   = cov(D4_BG);

% calculate SIGMAs (sigma) and their inverses
SIGMA_cheetah = cov_cheetah;
SIGMA_grass   = cov_grass;
SIGMAInv_cheetah = inv(SIGMA_cheetah);
SIGMAInv_grass   = inv(SIGMA_grass);

% calculate the coefficients
deno_cheetah = (sqrt(((2 * pi)^64) * det(SIGMA_cheetah)));
deno_grass   = (sqrt(((2 * pi)^64) * det(SIGMA_grass)));

% results
errorML_D4 = zeros(1, 9);  %TODO: change

for d = 1 : 1

    % calculate DCT2 (64D)
    error = 0;
    for i = 1 : m
        parfor j = 1 : n
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
            P_X_likelihood_cheetah = exp(-0.5 * (X - mu_cheetah) * (SIGMAInv_cheetah) * (X - mu_cheetah).') / deno_cheetah;
            P_X_likelihood_grass   = exp(-0.5 * (X - mu_grass) * (SIGMAInv_grass) * (X - mu_grass).') / deno_grass;
            P_X_cheetah = P_X_likelihood_cheetah * Prior_cheetah;
            P_X_grass   = P_X_likelihood_grass   * Prior_grass;
    
            % calculate the error
            if (P_X_grass > P_X_cheetah)
                result = 0;
            else
                result = 1;
            end
    
            % compare my result with the ideal mask
            if (result ~= cheetah_mask(i, j))
                error = error + 1;
            end    
        end
    end
    errorML_D4(1, d) = error / totalPixels;
end

for k = 2 : 9
    errorML_D4(1, k) = errorML_D4(1, 1);
end

save('errorML_D4.mat', 'errorML_D4')