clear
close all

fileName = 'field_data_spiral.mat';

load(fileName,'E_ret') % load etrieved Fields

sz = size(E_ret,1);
objNum = size(E_ret,3);

lambda = 0.532;
dx = (6.5*2)/(50*(100/180)*(100/200)*(250/200)); % pixel size

figure(1), imagesc(sum(abs(E_ret).^2,3)), axis image, title('Camera image'), colormap('gray')

%% Correlation Plane

z_list = 50:5:200;
metric_list = zeros(length(z_list),1);
dc_mask = ~mk_ellipse(5,5,size(E_ret,1),size(E_ret,2));

for ii = 1:length(z_list)

    I_sum = zeros(size(E_ret,[1,2]));

    for jj = 1:size(E_ret,3)-1

        test_id = setdiff(1:size(E_ret,3),jj);
        test = prop_ys(E_ret,lambda,z_list(ii),dx);

        I = conj(test(:,:,jj)).*test(:,:,test_id);
        I = mean(abs(IFFT(I)).^2,3);
        I = I.*dc_mask; % exclude auto-correlation

        I_sum = I_sum + I;
    end

    metric_list(ii) = max(I_sum(:));
end

[~,z_max_ind] = max(sum(metric_list,2));
dz = z_list(z_max_ind);

% figure(2), plot(z_list,metric_list), xlabel('propagation distance \mum'), ylabel('metric'), axis square, title(['Correlation plane at ', num2str(z_list(z_max_ind))])
disp(['Correlation plane found at ',num2str(dz)])

E_corr = prop_ys(E_ret,lambda,dz,dx); %field @ correlation plane

%%  Demixing 

E_corr_crop = E_corr(floor(sz/2+1)-floor(150/2):floor(sz/2+1)+ceil(150/2)-1,floor(sz/2+1)-floor(150/2):floor(sz/2+1)+ceil(150/2)-1,:);
E_corr_crop = E_corr_crop/sqrt(mean(abs(E_corr_crop).^2,'all'));


mu = 10^4.0;
uuMax = 300;

U_est = dlarray(RandomUnitary(objNum));

best_metric = 0;
demix_metric_list = zeros(uuMax,1);

disp('demixing starts')

for uu = 1:uuMax

    [demix_metric,dydx,objIter] = dlfeval(@(arg1,arg2,arg3) myCost(arg1,arg2,0), U_est, E_corr_crop);

    demix_metric_list(uu) = demix_metric;

    G = dydx*U_est'-U_est*dydx';
    G_size = 0.5*real(trace(extractdata(G*G')));

    P = expm(-mu*extractdata(G));

    while ~isfinite(P)
        mu = mu/(10^2);
        P = expm(-mu*extractdata(G));
    end

    P0 = P;

    Q = P*P;
    metric_Q = myCost(Q*U_est,E_corr_crop,1);
    if demix_metric-metric_Q >= mu*G_size
        P = Q;
        mu = 2*mu;
    end

    metric_P = myCost(P*U_est,E_corr_crop,1);
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

%% Virtual medium

E_demix = UnitaryTransform(E_corr,U_best);

recon_index = 1:objNum; 
skipInd = recon_index(1);

virtual_set = zeros(size(E_demix));
virtual_set(:,:,skipInd) = conj(E_demix(:,:,skipInd)); %%%%%%%%%

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

    for ii = 1:length(recon_index)

        if ~ismember(recon_index(ii),skipInd)
            compare_with = E_demix(:,:,recon_index(ii));
            comparison = abs(IFFT(diffuser_temp.*compare_with));

            maxVal = max(comparison(:));
            maxValList(recon_index(ii)) = maxVal;
        else
            continue
        end
    end

    [optVal,optInd] = max(maxValList);

    skipInd(end+1) = optInd;

    compare_with = E_demix(:,:,optInd);
    compare_with_pad = padarray(compare_with,[sz*x_pad_factor,sz*x_pad_factor],0,'both');
    diffuser_temp_pad = padarray(diffuser_temp,[sz*x_pad_factor,sz*x_pad_factor],0,'both'); 

    comparison = abs(IFFT(diffuser_temp_pad.*compare_with_pad));
    comparison = comparison/sqrt(mean(abs(diffuser_temp_pad(:)).^2)*mean(abs(compare_with_pad(:)).^2));

    [~,maxInd] = max(comparison(:));

    [dky,dkx] = ind2sub([sz*(2*x_pad_factor+1),sz*(2*x_pad_factor+1)], maxInd);
    dky = (dky - floor((2*x_pad_factor+1)*sz/2+1))/(2*x_pad_factor+1);
    dkx = (dkx - floor((2*x_pad_factor+1)*sz/2+1))/(2*x_pad_factor+1);

    ramp_term = exp(-1i*2*pi*( dkx*xx+dky*yy ));

    for pp = 1:phaseStep
        metric(pp) = sum(abs(diffuser_temp + conj(compare_with).*ramp_term.*exp(1i*phaseShift(pp))).^2,'all');
    end

    x = phaseMatrix\metric;
    virtual_set(:,:,optInd) = conj(compare_with).*ramp_term.*exp(1i*angle(x(3)));
end

virtual_est = sum(virtual_set,3);

figure(4), imagescX(exp(1i*angle(virtual_est))), axis image off, title('Scattering layer')

virtual_est = exp(1i*angle(virtual_est));

field_new = E_demix.*virtual_est;
field_new = padarray(field_new,[sz*(x_pad_factor),sz*(x_pad_factor)],0,'both');

beam_mask = mk_ellipse(75,75,size(field_new,1),size(field_new,2));
field_new = IFFT(field_new.*beam_mask);

obj_image = sum(abs(field_new).^2,3);

figure(5), imagesc(obj_image), axis image, title('Reconstructed image'), colormap('gray')


%% Custom function

function [val,dydx,objNew] = myCost(SU,field,flag)
    field_demix = dlarray(zeros(size(field)));
    for dd = 1:size(field,4)
        field_demix(:,:,:,dd) = UnitaryTransform(field(:,:,:,dd),SU);
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


function field_unitary = UnitaryTransform(field,SU)
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