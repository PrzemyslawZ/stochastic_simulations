clear all;
close all;

%%%%User-defined model parameters%%%%%
nt=10; %number of trajectories
N=10; %number of sites
hop_rate_list=[5];%hopping rate
gain_right_list=[5]; %rate of injection of the right site
loss_right_list=0*gain_right_list; %rate of dissipation of the right site
loss_left=1; %rate of dissipation of the left site
loss_chain=0; %loss rate on each chain site

tend=300;
dtgrid=1e-3;%regularized grid time step
ntgrid=floor(tend/dtgrid);%nb of grid time step
t_grid=linspace(0,tend,ntgrid);
navg=zeros(length(t_grid),N);%average chain configuration

%%%%%%Simulation%%%%%%
tic
    for p=1:length(hop_rate_list)
        for q=1:length(gain_right_list)
            [p q]
            gain_right=gain_right_list(p);
            loss_right=loss_right_list(p);
            hop_rate=hop_rate_list(q);
           
            for k=1:nt
                nreg=sim_hopping_MC(N,hop_rate,loss_left,loss_right,gain_right,loss_chain,t_grid);
                fprintf('size nreg %s\n', mat2str(size(nreg)));
                fprintf('size navg %s\n', mat2str(size(navg)));
                navg=navg+nreg; %
            end
            navg=navg./nt;

            %%%%%Saving the data%%%%%
            filename=strcat('data_gain',num2str(gain_right),'_hoprate',num2str(hop_rate));
            filename=regexprep(filename,'[.]','');%Remove dots in string
            save(filename,'navg','nt','hop_rate','gain_right','t_grid','N')
        end
    end
toc

%%%%%%Visualisation%%%%%%%
%figure(1)
%plot(t_grid,navg(:,end)) %Population of the last site versus time

figure(2)
plot(navg(end,:)) %Chain configuration at final time

