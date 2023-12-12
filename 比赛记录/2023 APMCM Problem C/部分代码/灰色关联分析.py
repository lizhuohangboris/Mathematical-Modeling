Matrix=input('请输入归一化后的数据:');
%提示输入归一化数据
19
[s,n]=size(Matrix);
%s 为样本数量，n 为序列数量
del = zeros(s,n-1);
for i = 1:n-1
del(:,i) = abs(Matrix(:,1) - Matrix(:,1+i));
% 序列差
end
r=0.5;
%分辨系数，一般情况下取 0.5
M = max(max(del));
%得两级最大差
m = min(min(del));
%得两级最小差
A = m + r * M;
B = r * M;
GCC = zeros(s,n-1);
for i = 1:n-1
GCC(:,i) = A ./ (B + del(:,i));
%由此得到各序列的灰色关联系数
end
GCD = mean(GCC);
%对灰色关联系数取均值得到灰色关联度
disp(GCD)