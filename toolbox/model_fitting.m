function [yi] = model_fitting(pre_x,pre_y,xi)
    % figure;
    warning off
    options=optimset('MaxFunEvals',1e6,'TolFun',1e-9,'TolX',1e-9, 'Display',  'off' );
    % disp(pre_x)
    % disp(pre_y)
    % disp(xi)
    offs=pre_x;
    offss = xi;
    zTemp = pre_y;
    % Fitted parameters
    % %                                 4. MT          
    % %      Zi    A1    G1   dw1     A4     G4    dw4      
    lb = [ 1       0.6   0.3   -1.0   0.05   10   -2.2    ];
    iv = [ 1       0.9   1.4   0      0.1    25    -2    ];
    ub = [ 1       1     10   +1.0    0.9    50    -1.8  ];
    remove_offs = [-3.9,-3.8,-3.7,-3.6,-3.5,-3.4,-3.3,-3.2,-3.1,     1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,   3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9];
    
    % 去除noe,cr,amide峰附近的点
    removed_offs = [];
    removed_zTemp = [];
    
    for i = 1:length(offs)
        % 使用容忍度 eps 进行比较
        % if all(abs(ones(1, length(remove_offs)) * offs(i) - remove_offs) > eps)
        if all(abs(offs(i) - remove_offs) > 1e-3)
            removed_offs = [removed_offs, offs(i)];
            removed_zTemp = [removed_zTemp, zTemp(i)];
        end
    end

    % pause(100)
    % disp(size(removed_offs));
    par = lsqcurvefit(@lorentzian2pool,iv,removed_offs,removed_zTemp,lb,ub,options);




    count =1;
    
    if mod(count, 10) == 0
                            % Fitted curves
        pn=2;
        res = zSpecLorentzDecomp(par, offss, pn);
        % disp(res);
        % disp("   ")
        zTempFit = res(:,1);
        waterTemp = res(:,2);
        mtTemp = res(:,3);

        % disp("mt--Cr")
        % disp(mtTemp);
        % disp(CrTemp);
    
       % figure;
       plot(removed_offs, removed_zTemp, 'bo',offss,zTempFit,'b-',offss, waterTemp, 'r-.', offss, mtTemp, 'm-.', 'LineWidth',0.5);
%                     axis([-10,10,0,1.02]); 
        axis([min(removed_offs(:)),max(removed_offs(:)),0,1.01]); 
        xlabel('Offset (ppm)'); ylabel('Z (%)');
        title('Z spectrum'); 
        legend('Z', 'Z_f_i_t', 'MT', 'Location', 'southeast');
        set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 14);
    end
    

    % 二次拟合
    pn = 5;
    % %             1. Water                  2. Amide               3. NOE                 4. MT                  5. Cr
    % %      Zi     A1    G1       dw1         A2     G2    dw2       A3     G3    dw3       A4      G4    dw4       A5     G5    dw5
    % lb = [ 1       par(2)*0.9  par(3)   par(4)      0.0025 0.4   3.2   0.00001  1.0   -3.7   par(5)*0.8 par(6) par(7)    0.001   0.4     1.6 ];
    % iv = [ 1       par(2)      par(3)   par(4)      0      1.0   3.5   0.02     5.0   -3.5   par(5)*0.9 par(6) par(7)    0.05    1.0     1.8];
    % ub = [ 1       par(2)      par(3)   par(4)      0.2    5.0   3.8   0.3      5.0   -3.3   par(5)*1.1 par(6) par(7)    0.1     1.5     2.0];
    lb = [ 1       par(2)*0.9  par(3)*0.9   par(4)      0.0025 0.4   3.2   0.00001  1.0   -3.7   par(5)*0.8 par(6)*0.9 par(7)    0.001   0.4     1.6 ];
    iv = [ 1       par(2)      par(3)       par(4)      0      1.0   3.5   0.02     5.0   -3.5   par(5)*0.9 par(6)     par(7)    0.05    1.0     1.8];
    ub = [ 1       par(2)      par(3)*1.1   par(4)      0.2    5.0   3.8   0.3      5.0   -3.3   par(5)*1.1 par(6)*1.1 par(7)    0.1     1.5     2.0];
    par = lsqcurvefit(@lorentzian5pool,iv,offs,zTemp,lb,ub,options);


    if mod(count, 10) == 0
        % Fitted curves
        res = zSpecLorentzDecomp(par, offss, pn);
        % disp(res);
        % disp("     ")
        zTempFit = res(:,1);
        waterTemp = res(:,2);
        amideTemp = res(:,3);
        noeTemp = res(:,4);
        mtTemp = res(:,5);
        CrTemp = res(:,6);

       % figure,
       plot(offs, zTemp, 'bo', offss, zTempFit, 'b-',...
        offss, waterTemp, 'r-.', offss, amideTemp, 'g-.',...
        offss, noeTemp, 'c-.', offss, mtTemp, 'm-.', offss, CrTemp, 'k-.', 'LineWidth',0.5);
%                     axis([-10,10,0,1.02]); 
        axis([min(offs(:)),max(offs(:)),0,1.01]); 
        xlabel('Offset (ppm)'); ylabel('Z (%)');
        title('Z spectrum'); 
        legend('Z', 'Z_f_i_t', 'Water', 'Amide', 'NOE', 'MT','Cr', 'Location', 'southeast');
        set(gca, 'Xdir', 'reverse', 'FontWeight', 'bold', 'FontSize', 14)

        % disp(par(12));
        % pause(0.1);
    end
    % pause(5);
    % close all;
    res = zSpecLorentzDecomp(par, offss, pn);
    zTempFit = res(:,1);
    yi =  squeeze(zTempFit(:));
end
