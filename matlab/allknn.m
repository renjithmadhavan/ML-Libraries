% Load the Data.
data = csvread('../data/fisheriris_data.csv');
% Choose the last two columns.
data = data(:,3:4);
% Contains query points.
query = [5 1.45;6 2;2.75 .75]

% Calculate the all 2-nearest-neighbors .
[n,d]=knnsearch(data,query,'K',2,'distance','euclidean', 'NSMethod', 'kdtree');

% Show the results.
d
n

% Plot the results
gscatter(data(:,1), data(:,2))
line(query(:,1), query(:,2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')
line(data(n,1), data(n,2),'color',[.5 .5 .5],'marker','o',...
   'linestyle','none','markersize',10)