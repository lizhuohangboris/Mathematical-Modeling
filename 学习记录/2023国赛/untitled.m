data=xlsread('各类每日销售量.xlsx','sheet2','B2:G1096');
x=[1:1095];
x=x';
y1=data(:,1);
y2=data(:,2);
y3=data(:,3);
y4=data(:,4);
y5=data(:,5);
y6=data(:,6);
figure(1)
hold on
plot(x,y1,'b', 'LineWidth', 1)
plot(x,y2,'r','LineWidth', 1)
plot(x,y3,'k','LineWidth', 1)
set(gca,'linewidth',1)
set(gca,'Box', 'on')
figure(2)
hold on
plot(x,y4,'g','LineWidth', 1)
plot(x,y5,'c','LineWidth', 1)
plot(x,y6,'m','LineWidth', 1)
36
set(gca,'linewidth',1)
set(gca,'Box', 'on')
data=xlsread('特征单品蔬菜每季度销售量.xlsx','sheet1','A2:L13');
x=[1:12];
x=x';
y1=data(:,1);
y2=data(:,2);
y3=data(:,3);
y4=data(:,4);
y5=data(:,5);
y6=data(:,6);
y7=data(:,7);
y8=data(:,8);
y9=data(:,9);
y10=data(:,10);
y11=data(:,11);
y12=data(:,12);
figure(1)
hold on
plot(x,y1,'b')
plot(x,y2,'r')
figure(2)
hold on
plot(x,y3,'k')
plot(x,y4,'g')
figure(3)
hold on
plot(x,y5,'c')
plot(x,y6,'m')
figure(4)
hold on
plot(x,y7,'b')
plot(x,y8,'r')
figure(5)
hold on
plot(x,y9,'k')
plot(x,y10,'g')
figure(6)
hold on
plot(x,y11,'c')
plot(x,y12,'m')
