% This script reconstructs images using the retrieved fields (E_ret).
% The field retrieval step is skipped here.
% For the field retrieval, refer to phase_retrieval_example.m.
%
% Requires Matlab 2022b or later due to dlarray.
% Written by YoonSeok Baek

%% Initialization
% load retrieved fields and define parameters

clear
close all

dataName = 'field_data_spiral.mat'; 

script_dir = fileparts(matlab.desktop.editor.getActiveFilename);
cd(script_dir);

if ~isfile(dataName)
    disp('Cannot find the data file. Open it manually')
    [~,dataPath] = uigetfile('*.mat','open field_data_spiral.mat');
    cd(dataPath)
end

load(dataName,'E_ret') % load retrieved fields (E_ret)

sz = size(E_ret,1); 
objNum = size(E_ret,3); % number of incoherent sources

lambda = 0.532; % wavelength of light in um
dx = (6.5*2)/(50*(100/180)*(100/200)*(250/200)); % pixel size at the object plane in the experimental setup.

%% Find Correlation Plane
% To find the correlation plane using a metric; see the Method section (Locating the correlation plane)

z_list = 50:5:200; % search range for the correlation plane
metric_list = zeros(length(z_list),1);
dc_mask = ~mk_ellipse(5,5,size(E_ret,1),size(E_ret,2)); % mask to exclude self-correlation components

for ii = 1:length(z_list)
    I_sum = zeros(size(E_ret,[1,2]));

    for jj = 1:size(E_ret,3)-1

        test_id = setdiff(1:size(E_ret,3),jj);
        test = propagation_spectral(E_ret,lambda,z_list(ii),dx);

        I = conj(test(:,:,jj)).*test(:,:,test_id);
        I = mean(abs(IFFT(I)).^2,3); 
        I = I.*dc_mask;

        I_sum = I_sum + I;
    end

    metric_list(ii) = max(I_sum(:));
end

[~,z_max_ind] = max(sum(metric_list,2));
dz = z_list(z_max_ind); % distance to propagate for the correlation plane

% figure(2), plot(z_list,metric_list), xlabel('propagation distance \mum'), ylabel('metric'), axis square, title(['Correlation plane at ', num2str(z_list(z_max_ind))])
disp(['Correlation plane found at ',num2str(dz), ' um'])

E_corr = propagation_spectral(E_ret,lambda,dz,dx); % fields at the correlation plane


%% Demixing Retrieved Fields (E_ret)
% To unitary transform fields; see the Method section (Demixing fields for individual sources)
% see also [Abrudan, T. E., Eriksson, J., & Koivunen, V. (2008). Steepest descent algorithms for optimization under unitary matrix constraint. IEEE Transactions on Signal Processing, 56(3), 1134-1147.]

E_corr_crop = E_corr(floor(sz/2+1)-floor(150/2):floor(sz/2+1)+ceil(150/2)-1,floor(sz/2+1)-floor(150/2):floor(sz/2+1)+ceil(150/2)-1,:);
E_corr_crop = E_corr_crop/sqrt(mean(abs(E_corr_crop).^2,'all'));

mu = 10^4.0; % 
uuMax = 300; % Maximum iteration

U_est = dlarray(RandomUnitary(objNum));

best_metric = 0;
demix_metric_list = zeros(uuMax,1);

disp('demixing starts')

for uu = 1:uuMax

    [demix_metric,dydx,objIter] = dlfeval(@(arg1,arg2,arg3) demixing_cost(arg1,arg2,0), U_est, E_corr_crop);

    demix_metric_list(uu) = demix_metric;

    G = dydx*U_est'-U_est*dydx';
    G_size = 0.5*real(trace(extractdata(G*G')));

    P = expm(-mu*extractdata(G));

    while ~isfinite(P) % prevent divergence
        mu = mu/(10^2);
        P = expm(-mu*extractdata(G));
    end

    P0 = P;

    Q = P*P;
    metric_Q = demixing_cost(Q*U_est,E_corr_crop,1);
    if demix_metric-metric_Q >= mu*G_size
        P = Q;
        mu = 2*mu;
    end

    metric_P = demixing_cost(P*U_est,E_corr_crop,1);
    if demix_metric-metric_P < 0.5*mu*G_size
        P = P0;
        mu = 0.5*mu;
    end

    U_est = P*U_est;

    if uu>10 && abs(demix_metric_list(uu)-demix_metric_list(uu-1)) < 10^-5
        demix_metric_list = demix_metric_list(1:uu);
        break
    end

end

if min(demix_metric_list(:)) < best_metric
    U_best = extractdata(U_est);
    best_metric = min(demix_metric_list(:));
end

disp('demixing finished')
% figure(3), plot(imMetricList), axis square, title('Demixing iteration')

E_demix = unitaryTransform(E_corr,U_best); % demixed fields after the unitary transformation 

%% Virtual Medium Construction

recon_index = 1:objNum; 
skipInd = recon_index(1); % reference index, equivalent to the index "m" of "E_m" in the main text 

virtual_set = zeros(size(E_demix));
virtual_set(:,:,skipInd) = E_demix(:,:,skipInd);

phaseStep = 3;
phaseShift = ((1:phaseStep)-1)/phaseStep*(2*pi);
phaseMatrix = single([ones(phaseStep, 1), exp(1i * phaseShift'), exp(-1i * phaseShift')]);
metric = zeros(phaseStep,1);

x_pad_factor = 2;

[yy,xx] = ndgrid((1:sz)-floor(sz/2+1),(1:sz)-floor(sz/2+1));
xx = xx/sz;
yy = yy/sz;

for jj = 1:length(recon_index)-1 

    maxValList = NaN(objNum,1);
    diffuser_temp = sum(virtual_set,3);

    % find a field with the maximum correlation with the current virtual medium.
    for ii = 1:length(recon_index)

        if ~ismember(recon_index(ii),skipInd)
            compare_with = E_demix(:,:,recon_index(ii));
            comparison = abs(FFT(diffuser_temp.*conj(compare_with)));

            maxVal = max(comparison(:));
            maxValList(recon_index(ii)) = maxVal;
        else
            continue
        end
    end

    [optVal,optInd] = max(maxValList);
    skipInd(end+1) = optInd; %#ok<SAGROW>

    % update the virtual meidum using the field with the maximum correlation.
    % (1) identify the relative phase ramp.
    compare_with = E_demix(:,:,optInd);
    compare_with_pad = padarray(compare_with,[sz*x_pad_factor,sz*x_pad_factor],0,'both');
    diffuser_temp_pad = padarray(diffuser_temp,[sz*x_pad_factor,sz*x_pad_factor],0,'both'); 

    comparison = abs(FFT(diffuser_temp_pad.*conj(compare_with_pad)));
    comparison = comparison/sqrt(mean(abs(diffuser_temp_pad(:)).^2)*mean(abs(compare_with_pad(:)).^2));

    [~,maxInd] = max(comparison(:));

    [dky,dkx] = ind2sub([sz*(2*x_pad_factor+1),sz*(2*x_pad_factor+1)], maxInd);
    dky = (dky - floor((2*x_pad_factor+1)*sz/2+1))/(2*x_pad_factor+1);
    dkx = (dkx - floor((2*x_pad_factor+1)*sz/2+1))/(2*x_pad_factor+1);

    ramp_term = exp(1i*2*pi*( dkx*xx+dky*yy ));

    % (2) adjust constant phase before updating the virtual medium
    for pp = 1:phaseStep
        metric(pp) = sum(abs(diffuser_temp + compare_with.*ramp_term.*exp(1i*phaseShift(pp))).^2,'all');
    end
    x = phaseMatrix\metric;

    % (3) save the phase-ramp-correted field.
    virtual_set(:,:,optInd) = compare_with.*ramp_term.*exp(1i*angle(x(3)));
end

virtual_medium = sum(virtual_set,3); 
virtual_medium = exp(1i*angle(virtual_medium));

figure(4), imagesc(angle(virtual_medium)), axis image, title('Virtual scattering layer'), colorbar

%% Image Reconstruction

field_corrected = conj(E_demix).*virtual_medium;
field_corrected = padarray(field_corrected,[sz*(x_pad_factor),sz*(x_pad_factor)],0,'both');

beam_mask = mk_ellipse(75,75,size(field_corrected,1),size(field_corrected,2)); % to exclude non-physical noise
field_corrected = FFT(field_corrected.*beam_mask);

obj_image = sum(abs(field_corrected).^2,3);

figure(5), imagesc(obj_image), axis image, title('Reconstructed image'), colormap('gray')


%% Custom Functions

function [val,dydx,objNew] = demixing_cost(SU,field,flag)
    field_demix = dlarray(zeros(size(field)));
    for dd = 1:size(field,4)
        field_demix(:,:,:,dd) = unitaryTransform(field(:,:,:,dd),SU);
    end

    if flag == 0
        imTemp = dlarray(zeros(size(field,1),size(field,2),size(field,3)-1));
    else
        imTemp = (zeros(size(field,1),size(field,2),size(field,3)-1));
        dydx = [];
    end

    field_num = size(field,3);
    dc_mask = ~mk_ellipse(0.5,0.5,size(field,1),size(field,2));

    for nn = 1:1:field_num
        ind_temp = 1:field_num; ind_temp = setdiff(ind_temp,nn);

        if length(ind_temp) == 1 && size(field,4) == 1
            imTemp2 = conj(field_demix(:,:,nn,:)).*field_demix(:,:,ind_temp,:);

            imTemp2 = fftshift(fft(ifftshift(imTemp2)));
            imTemp2 = fftshift(fft(ifftshift(imTemp2.'))).';
        else
            imTemp2 = FFT(conj(field_demix(:,:,nn,:)).*field_demix(:,:,ind_temp,:));
        end

        imTemp2 = abs(imTemp2);
        imTemp2 = imTemp2.*dc_mask;
        imTemp(nn) = mean(max(imTemp2,[],[1,2]),'all');
    end
    val = -mean(imTemp(:));

    if flag == 0
        dydx = dlgradient(val,SU);
    end

    objNew = [];
end

function U_rand = RandomUnitary(U_size)
    X = single((randn(U_size) + 1i*randn(U_size))/sqrt(2));
    [Q,R] = qr(X);
    R = diag(diag(R)./abs(diag(R)));
    U_rand = Q*R;
end


function field_unitary = unitaryTransform(field,SU)
    field_unitary = reshape(field,[],size(field,3));
    field_unitary = field_unitary*SU;
    field_unitary = reshape(field_unitary,size(field,1),size(field,2),size(field,3));
end

function ellipseMask = mk_ellipse(yr,xr,sy,sx)
    [yy,xx] = ndgrid((1:sy)-floor(sy/2+1),(1:sx)-floor(sx/2+1));
    ellipseMask = (xx/xr).^2+(yy/yr).^2<=1;
end

function out = FFT(in)
    out = fftshift(fft2(ifftshift(in)));
end

function out = IFFT(in)
    out = fftshift(ifft2(ifftshift(in)));
end

function E_out = propagation_spectral(E_in,wavelength,propagation_distance,pixel_size)
    % propagate field according to the spectral method
    if propagation_distance ~= 0
    
        Ny = size(E_in,1);
        Nx = size(E_in,2);
    
        dkx = 1/(Nx*pixel_size); % k-space pixel size
        dky = 1/(Ny*pixel_size);
    
        [ky,kx] = ndgrid((1:Ny)-floor(Ny/2+1),(1:Nx)-floor(Nx/2+1));
        kx = kx*dkx;
        ky = ky*dky;
    
        kr_square = kx.^2+ky.^2;
    
        defocus = ((1/wavelength)^2>=kr_square).*exp( 2i*pi*propagation_distance*real(sqrt((1/wavelength)^2-kr_square)));
        E_out = IFFT( defocus .* FFT(E_in) );
    
    else
        E_out = E_in;
    end

end
