function GA()
clear; close all; clc;

% 定义被控系统传递函数
a = tf([5.1691, 522.4673, 8184.2371], [0.0002, 0.5182, 18.2593, 200.4213, 846.3742, 1618.3576, 6547.3869]);

% 遗传算法参数
popsize = 25;       % 增加种群大小
chromlength = 60;    
pc = 0.6;           
pm = 0.15;          % 提高变异概率
G = 50;             % 增加迭代次数

% PID参数范围 - 特别注意限制Ki范围
ranges = [0, 0.1;    % Kp范围 (降低上限)
          0, 5;     % Ki范围 (大幅降低上限)
          0, 1.5];   % Kd范围

% 初始化种群
pop = round(rand(popsize, chromlength)); 
J_history = zeros(1, G);
best_params = zeros(G, 3);

% 初始种群评估
decpop = bintodec(pop, popsize, chromlength, ranges);
fx = calobjvalue(decpop, a);
[best_J, idx] = min(fx);
best_params(1,:) = decpop(idx,:);
J_history(1) = best_J;

% 创建图形窗口
figure;
set(gcf, 'Position', [100, 100, 1400, 600]);

% 主循环
for i = 2:G
    % 计算适应度
    decpop = bintodec(pop, popsize, chromlength, ranges);
    fx = calobjvalue(decpop, a);
    fitvalue = calfitvalue(fx);
    
    % 遗传操作
    newpop = copyx(pop, fitvalue, popsize);
    newpop = crossover(newpop, pc, popsize, chromlength);
    newpop = mutation(newpop, pm, popsize, chromlength);
    
    % 评估新种群
    new_decpop = bintodec(newpop, popsize, chromlength, ranges);
    new_fx = calobjvalue(new_decpop, a);
    new_fitvalue = calfitvalue(new_fx);
    
    % 更新种群（精英保留）
    index = new_fitvalue > fitvalue;
    pop(index, :) = newpop(index, :);
    
    % 评估当前种群
    decpop = bintodec(pop, popsize, chromlength, ranges);
    fx = calobjvalue(decpop, a);
    [best_J, idx] = min(fx);
    J_history(i) = best_J;
    best_params(i,:) = decpop(idx,:);
    
    % 绘制结果
    plot_results(a, best_params(i,:), J_history(1:i), i, G);
end

% 输出最终结果
[opt_J, gen] = min(J_history);
opt_params = best_params(gen,:);
fprintf('最优参数: Kp=%.4f, Ki=%.4f, Kd=%.4f\n', opt_params(1), opt_params(2), opt_params(3));
fprintf('最小性能指标: %.4f\n', opt_J);

% 绘制最终响应曲线
figure;
plot_final_response(a, opt_params);
end

%% 改进的目标函数（加入振荡惩罚）
function fx = calobjvalue(decpop, sys)
    fx = zeros(1, size(decpop,1));
    for i = 1:size(decpop,1)
        Kp = decpop(i,1);
        Ki = decpop(i,2);
        Kd = decpop(i,3);
        
        % 创建PID控制器
        C = pid(Kp, Ki, Kd);
        
        % 构建闭环系统
        closed_loop = feedback(series(C, sys), 1);
        
        try
            % 延长仿真时间到20秒
            [y, t] = step(closed_loop, 0:0.01:20);
            e = 1 - y;  % 计算误差
            
            % 计算主要性能指标
            ISE = sum(e.^2) * 0.01;
            
            % 1. 检测振荡并添加惩罚项
            steady_state_index = t > 5; % 5秒后的稳态部分
            y_steady = y(steady_state_index);
            
            % 计算稳态振荡幅度
            oscillation_amp = 0.5 * (max(y_steady) - min(y_steady));
            
            % 2. 检查稳定性（相位裕度）
            [Gm, Pm] = margin(series(C, sys));
            stability_penalty = 0;
            if Pm < 30  % 相位裕度小于30度视为不稳定
                stability_penalty = 1000 * (30 - Pm);
            end
            
            % 3. 组合性能指标
            % 基础ISE + 振荡惩罚 + 稳定性惩罚
            fx(i) = ISE + 50 * oscillation_amp + stability_penalty;
            
        catch
            % 处理不稳定系统
            fx(i) = 1e10;
        end
    end
end

%% 二进制转十进制（支持多参数）
function decpop = bintodec(pop, popsize, chromlength, ranges)
    seg_len = chromlength / 3; 
    decpop = zeros(popsize, 3); 
    
    for i = 1:popsize
        % Kp转换
        bin_part = pop(i, 1:seg_len);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        decpop(i,1) = ranges(1,1) + dec_val * (ranges(1,2)-ranges(1,1)) / (2^seg_len-1);
        
        % Ki转换（应用非线性缩放，使小值更精确）
        bin_part = pop(i, seg_len+1:2*seg_len);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        % 非线性映射：优先小值
        scaled_value = dec_val / (2^seg_len-1);
        nonlinear_value = scaled_value^2; % 平方映射使小值更敏感
        decpop(i,2) = ranges(2,1) + nonlinear_value * (ranges(2,2)-ranges(2,1));
        
        % Kd转换
        bin_part = pop(i, 2*seg_len+1:end);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        decpop(i,3) = ranges(3,1) + dec_val * (ranges(3,2)-ranges(3,1)) / (2^seg_len-1);
    end
end

%% 绘制迭代结果（增强版）
function plot_results(sys, params, J_history, gen, max_gen)
    Kp = params(1);
    Ki = params(2);
    Kd = params(3);
    
    % 创建控制器和闭环系统
    C = pid(Kp, Ki, Kd);
    closed_loop = feedback(series(C, sys), 1);
    
    % 延长仿真时间
    [y, t] = step(closed_loop, 0:0.01:20);
    
    % 创建双图布局
    subplot(1,3,1);
    plot(t, y, 'LineWidth', 1.8);
    hold on;
    plot([t(1), t(end)], [1, 1], 'r--', 'LineWidth', 1.5);
    
    % 标记振荡区域
    steady_region = t > 5;
    if any(steady_region)
        plot(t(steady_region), y(steady_region), 'g', 'LineWidth', 2.5);
        oscillation_amp = 0.5*(max(y(steady_region))-min(y(steady_region)));
        title(sprintf('响应曲线 (振荡: %.4f)', oscillation_amp));
    else
        title('响应曲线');
    end
    
    hold off;
    xlabel('时间 (s)');
    ylabel('响应');
    grid on;
    axis([0, 20, 0, 2.0]);
    legend('全响应', '期望值', '稳态区', 'Location', 'Southeast');
    
    % 绘制性能指标进化
    subplot(1,3,2);
    plot(1:gen, J_history, 'b-o', 'LineWidth', 1.8, 'MarkerFaceColor', 'b');
    title('性能指标进化');
    xlabel('迭代次数');
    ylabel('综合性能指标');
    grid on;
    axis tight;
    
    % 添加参数变化曲线
    subplot(1,3,3);
    hold on;
    plot(1:gen, best_params(1:gen,1), 'r-o', 'LineWidth', 1.5, 'DisplayName', 'Kp');
    plot(1:gen, best_params(1:gen,2), 'g-s', 'LineWidth', 1.5, 'DisplayName', 'Ki');
    plot(1:gen, best_params(1:gen,3), 'b-^', 'LineWidth', 1.5, 'DisplayName', 'Kd');
    hold off;
    title('PID参数进化');
    xlabel('迭代次数');
    ylabel('参数值');
    legend show;
    grid on;
    axis tight;
    
    % 添加整体标题
    sgtitle(sprintf('第 %d/%d 代: Kp=%.4f, Ki=%.4f, Kd=%.4f', gen, max_gen, Kp, Ki, Kd), 'FontSize', 14);
    
    drawnow;
end

%% 绘制最终响应曲线（增强版）
function plot_final_response(sys, params)
    % 创建控制器和闭环系统
    C = pid(params(1), params(2), params(3));
    closed_loop = feedback(series(C, sys), 1);
    
    % 延长仿真时间
    [y, t] = step(closed_loop, 0:0.01:20);
    
    % 计算性能指标
    e = 1 - y;
    ISE = sum(e.^2) * 0.01;
    IAE = sum(abs(e)) * 0.01;
    ITAE = sum(t.*abs(e)) * 0.01;
    
    % 检测稳态振荡
    steady_state_index = t > 5;
    y_steady = y(steady_state_index);
    oscillation_amp = 0.5 * (max(y_steady) - min(y_steady));
    
    % 创建双图布局
    figure('Position', [100, 100, 1200, 500]);
    
    % 响应曲线
    subplot(1,2,1);
    plot(t, y, 'LineWidth', 2.2);
    hold on;
    plot([t(1), t(end)], [1, 1], 'r--', 'LineWidth', 1.8);
    
    % 标记稳态区域
    if any(steady_state_index)
        plot(t(steady_state_index), y(steady_state_index), 'g', 'LineWidth', 2.8);
        text(12, 0.8, sprintf('振荡幅度: %.4f', oscillation_amp), 'FontSize', 12);
    end
    
    hold off;
    title(sprintf('PID控制响应: Kp=%.4f, Ki=%.4f, Kd=%.4f', params(1), params(2), params(3)));
    xlabel('时间 (s)');
    ylabel('响应');
    grid on;
    axis([0, 20, 0, 1.8]);
    legend('系统响应', '期望值', '稳态区', 'Location', 'Southeast');
    
    % 性能指标
    subplot(1,2,2);
    metrics = [ISE, IAE, ITAE, oscillation_amp];
    bar(metrics, 'FaceColor', [0.2, 0.6, 0.8]);
    set(gca, 'XTickLabel', {'ISE', 'IAE', 'ITAE', 'Osc Amp'});
    title('性能指标比较');
    ylabel('指标值');
    grid on;
    
    % 添加整体标题
    sgtitle(sprintf('最优PID控制性能 (振荡幅度: %.4f)', oscillation_amp), 'FontSize', 14);
end