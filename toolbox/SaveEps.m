function SaveEps(dir,name)

% set(0,'CurrentFigure',gcf);
% set(gcf,'PaperPositionMode','auto');
% set(gca,'position',[0,0,1,1]);
% set(gcf,'position',[50,50,width,height]);
mymkdir(dir);
name = strrep(name,'\',' ');
name = strrep(name,'/',' ');
saveas(gcf,[dir,filesep,name,'.eps'],'epsc');
saveas(gcf,[dir,filesep,name,'.emf'],'meta');
saveas(gcf,[dir,filesep,name,'.bmp'],'bmp');
saveas(gcf,[dir,filesep,name,'.png'],'png');
saveas(gcf,[dir,filesep,name,'.fig'],'fig');
saveas(gcf,[dir,filesep,name,'.tif'],'tiff');
saveas(gcf,[dir,filesep,name,'.pdf'],'pdf');
end