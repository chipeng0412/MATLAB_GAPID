%a=tf([5.1691 522.4673 8184.2371],[0.0002 0.5182 18.2593 200.4213 846.3742 1618.3576 6547.3869]);
%rlocus(a)
%%主程序
function GA()
clear; close all; clc;

% 定义被控系统传递函数
a = tf([5.1691, 522.4673, 8184.2371], [0.0002, 0.5182, 18.2593, 200.4213, 846.3742, 1618.3576, 6547.3869]);

% 遗传算法参数
popsize = 40;       % 种群大小
chromlength = 120;    % 染色体总长度（40位×3参数）
pc = 0.6;           % 交叉概率
pm = 0.1;           % 变异概率
G = 30;             % 迭代次数

% PID参数范围 [Kp_min, Kp_max; Ki_min, Ki_max; Kd_min, Kd_max]
ranges = [0, 1.5;    % Kp范围
          0, 10;    % Ki范围
          0, 5];   % Kd范围

% 初始化种群
pop = round(rand(popsize, chromlength)); 
J_history = zeros(1, G);  % 存储每代最小性能指标
best_params = zeros(G, 3); % 存储每代最优参数[Kp, Ki, Kd]

% 初始种群评估
decpop = bintodec(pop, popsize, chromlength, ranges);
fx = calobjvalue(decpop, a);
[best_J, idx] = min(fx);
best_params(1,:) = decpop(idx,:);
J_history(1) = best_J;

% 创建图形窗口
figure;
set(gcf, 'Position', [100, 100, 1200, 500]);

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
fprintf('最小性能指标(ISE): %.12f\n', opt_J);

% 绘制最终响应曲线
figure;
plot_final_response(a, opt_params);
end

%% 二进制转十进制（支持多参数）
function decpop = bintodec(pop, popsize, chromlength, ranges)
    seg_len = chromlength / 3;  % 每段长度
    decpop = zeros(popsize, 3); % [Kp, Ki, Kd]
    
    for i = 1:popsize
        % 转换Kp
        bin_part = pop(i, 1:seg_len);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        decpop(i,1) = ranges(1,1) + dec_val * (ranges(1,2)-ranges(1,1)) / (2^seg_len-1);
        
        % 转换Ki
        bin_part = pop(i, seg_len+1:2*seg_len);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        decpop(i,2) = ranges(2,1) + dec_val * (ranges(2,2)-ranges(2,1)) / (2^seg_len-1);
        
        % 转换Kd
        bin_part = pop(i, 2*seg_len+1:end);
        dec_val = sum(bin_part .* (2.^(seg_len-1:-1:0)));
        decpop(i,3) = ranges(3,1) + dec_val * (ranges(3,2)-ranges(3,1)) / (2^seg_len-1);
    end
end

%% 计算目标函数（性能指标）
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
            % 仿真阶跃响应
            [y, t] = step(closed_loop, 0:0.01:20);
            e = 1 - y;  % 计算误差
            
            % 计算ISE（积分平方误差）
            %ISE = sum(e.^2) * 0.01;  % 矩形法数值积分
            ITAE = sum(t.*abs(e)) * 0.01;
            fx(i) = ITAE;
        catch
            % 处理不稳定系统
            fx(i) = 1e10;
        end
    end
end

%% 计算适应度
function fitvalue = calfitvalue(fx)
    % 使用倒数形式（最小化问题转为最大化问题）
    fitvalue = 1./(fx + 1e-6);  % 加小常数避免除零
end

%% 绘制迭代结果
function plot_results(sys, params, J_history, gen, max_gen)
    % 获取当前最优PID参数
    Kp = params(1);
    Ki = params(2);
    Kd = params(3);
    
    % 创建控制器和闭环系统
    C = pid(Kp, Ki, Kd);
    closed_loop = feedback(series(C, sys), 1);
    
    % 仿真阶跃响应
    [y, t] = step(closed_loop, 0:0.01:20);
    
    % 绘制阶跃响应
    subplot(1,2,1);
    plot(t, y, 'LineWidth', 1.5);
    hold on;
    plot([t(1), t(end)], [1, 1], 'r--', 'LineWidth', 1.2);
    hold off;
    title(sprintf('第 %d/%d 代: Kp=%.8f, Ki=%.8f, Kd=%.8f', gen, max_gen, Kp, Ki, Kd));
    xlabel('时间 (s)');
    ylabel('响应');
    grid on;
    axis([0, 10, 0, 1.8]);
    legend('系统响应', '期望值', 'Location', 'Southeast');
    
    % 绘制性能指标进化
    subplot(1,2,2);
    plot(1:gen, J_history, 'b-o', 'LineWidth', 1.5);
    title('性能指标 (ITAE) 进化');
    xlabel('迭代次数');
    ylabel('ITAE');
    grid on;
    axis([1, max_gen, 0, max(J_history)*1.1]);
    
    drawnow;
end

%% 绘制最终响应曲线
function plot_final_response(sys, params)
    % 创建控制器和闭环系统
    C = pid(params(1), params(2), params(3));
    closed_loop = feedback(series(C, sys), 1);
    
    % 仿真阶跃响应
    [y, t] = step(closed_loop, 0:0.01:20);
    
    % 计算性能指标
    e = 1 - y;
    ISE = sum(e.^2) * 0.01;
    IAE = sum(abs(e)) * 0.01;
    ITAE = sum(t.*abs(e)) * 0.01;
    
    % 绘制响应曲线
    plot(t, y, 'LineWidth', 2);
    hold on;
    plot([t(1), t(end)], [1, 1], 'r--', 'LineWidth', 1.5);
    hold off;
    
    title(sprintf('最优PID控制: Kp=%.12f, Ki=%.12f, Kd=%.12f', params(1), params(2), params(3)));
    subtitle(sprintf('ISE=%.4f, IAE=%.4f, ITAE=%.4f', ISE, IAE, ITAE));
    xlabel('时间 (s)');
    ylabel('响应');
    grid on;
    axis([0, 10, 0, 1.8]);
    legend('系统响应', '期望值', 'Location', 'Southeast');
end

% ---------------------- 以下遗传操作函数保持不变 ----------------------

%% 复制操作
function newx = copyx(pop, fitvalue, popsize)
    newx = pop;
    p = fitvalue / sum(fitvalue);
    Cs = cumsum(p);
    R = sort(rand(popsize,1));
    j = 1;
    for i = 1:popsize
        while R(i) > Cs(j)
            j = j + 1;
        end
        newx(i,:) = pop(j,:);
    end
end

%% 交叉操作
function newx = crossover(pop, pc, popsize, chromlength)
    newx = pop;
    i = 2;
    while i <= popsize-1
        if rand < pc
            % 选择交叉点
            points = sort(randperm(chromlength, 2));
            start_idx = points(1);
            end_idx = points(2);
            
            % 执行交叉
            temp = newx(i-1, start_idx:end_idx);
            newx(i-1, start_idx:end_idx) = newx(i, start_idx:end_idx);
            newx(i, start_idx:end_idx) = temp;
        end
        i = i + 2;
    end
end

%% 变异操作
function newx = mutation(pop, pm, popsize, chromlength)
    newx = pop;
    for i = 1:popsize
        if rand < pm
            % 随机选择变异位
            idx = randi(chromlength);
            newx(i, idx) = ~newx(i, idx);
        end
    end
end







