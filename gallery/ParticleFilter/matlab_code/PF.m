clear all;
clf;
clc;

% Set random seed for repeatability
rng(1);

% Load data
load('dataset2.mat');

% Plot settings
blue = [0,0.4470,0.7410];
green = [0.4660,0.6740,0.1880];
grey = [0.8,0.8,0.8];
red = [0.6350,0.0780,0.1840];

%% Initialize video

vid = VideoWriter('pf.avi');
open(vid);
figure(1)
axis equal
landmark = l;
for i = 1:1:length(landmark)
    plot(landmark(i,1),landmark(i,2),'k.','markersize',15);
    landmarkLabel = sprintf('Landmark %d',i);
    text(landmark(i,1)-250,landmark(i,2)+100,landmarkLabel);
    hold all;
end
% axis([-2000,2000,-1500,1500]);
title('Experimental Setup');
xlabel('x [m]');
ylabel('y [m]');
window = [get(gca,'xlim'),get(gca,'ylim')];
% M = getframe;
% writeVideo(vid,M);

%% Paramters
% Particle filter parameters
nparticles = 100;               % number of particles
v_noise = sqrt(v_var);          % noise on longitudinal speed for propagating particle
omega_noise = sqrt(om_var);     % noise on rotational speed for propagating particle
laser_r_var = 0.5;   
laser_b_var = b_var;            % bearing variance
w_gain = 10;                    % gain on particle weight
r_max = 3;                      % max range measurement considered valid
w_threshold = 1e6;

%% Initialize particle filter
x_particle = x_true(1) + .8*randn(nparticles,1);
y_particle = y_true(1) + .8*randn(nparticles,1);
theta_particle = th_true(1) + 0.1*randn(nparticles,1);


x_estimate = mean(x_particle);
y_estimate = mean(y_particle);
th_estimate = mean(theta_particle);


%% Particle filter algorithm

% error variables for final error plots - set the errors to zero at the start
pf_err(1) = 0;
pf_x_err(1) = 0;
pf_y_err(1) = 0;
pf_th_err(1) = 0;
sum_weight(1) = 0;

% loop over laser scans
for i=2:size(t,1)
	% plotting
    figure(1);
% 	waitforbuttonpress;
    clf;
    hold on;
    landmark = l;
    for k = 1:1:length(landmark)
        plot(landmark(k,1),landmark(k,2),'k.','markersize',15);
    end
    title('Experimental Setup');
    xlabel('x [m]');
    ylabel('y [m]');
    % update the wheel-odometry-only algorithm
%     dt = t(i) - t(i-1);
    dt = 0.1;

	% Determine number of properly detected landmarks
    % Number of valid range measurements 
    % (number of landmarks 'properly' detected)
    valid_flag = (r(i,:) > 0) .* (r(i,:) < r_max);    
    valid_index = find(valid_flag==1);
    N_l = sum(valid_flag);
    
    if N_l > 0
        % loop over the particles
        for n=1:nparticles

            % Propagate the particle forward in time using wheel odometry
            % (remember to add some unique noise to each particle so they
            % spread out over time)
            % Sample velocity
            v_sample = v(i) + v_noise*randn(1);
            u_sample = v_noise*randn(1);
            omega_sample = om(i) + omega_noise*randn(1);

            % Updated pose
            x_particle(n) = x_particle(n) + dt*(v_sample*cos( theta_particle(n) ) - u_sample*sin( theta_particle(n) ));
            y_particle(n) = y_particle(n) + dt*(v_sample*sin( theta_particle(n) ) + u_sample*cos( theta_particle(n) ));
            phi = theta_particle(n) + dt*omega_sample;
            while phi > pi
                phi = phi - 2*pi;
            end
            while phi < -pi
                phi = phi + 2*pi;
            end 
            theta_particle(n) = phi;

            % Initialize weight of the particle (this is reset for each timestamp)
            w_particle(n) = 1.0;
            for j=1:1:N_l
                % For each particle compute weight (this loops thru observed landmarks)
                landmarkIndex = valid_index(j);
                landmarkPosition = l(landmarkIndex,:)';
                y_measured = [r(i,landmarkIndex);b(i,landmarkIndex)];
                y_est = g([x_particle(n);y_particle(n);theta_particle(n)],landmarkPosition,d,[0;0]);

                w_particle(n) = w_particle(n)*w_gain...
                    *exp(-0.5*(y_est(1) - y_measured(1))^2/(laser_r_var));
            end
        end
        sum_weight(i) = sum(w_particle);
        fprintf('time %.2f (i = %d); particle filter; sum(w) = %f\n',t(i),i,sum(w_particle));

        % Resample the particles using Madow systematic resampling
        w_bounds = cumsum(w_particle)/sum(w_particle);
        w_target = rand(1);
        j = 1;

        for n=1:nparticles
           while w_bounds(j) < w_target
               j = mod(j,nparticles) + 1;
           end
           x_particle_new(n) = x_particle(j);
           y_particle_new(n) = y_particle(j);
           theta_particle_new(n) = theta_particle(j);
           w_target = w_target + 1/nparticles;
           if w_target > 1
               w_target = w_target - 1.0;
               j = 1;
           end
        end
        

        
        hold all;
        x_particle = x_particle_new;
        y_particle = y_particle_new;
        theta_particle = theta_particle_new;        
        
        plot(x_particle,y_particle,'.','color',blue);

        % save the translational error for later plotting
        x_estimate = mean(x_particle);
        y_estimate = mean(y_particle);
        th_estimate = mean(theta_particle);
        
    else
        fprintf('time %.2f; odmometry\n',t(i));
        state_estimate = f([x_estimate;y_estimate;th_estimate],dt,[v(i);om(i)],[0;0]);
        x_estimate = state_estimate(1);
        y_estimate = state_estimate(2);
        th_estimate = state_estimate(3);
    end

    % Compute error
    pf_err(i) = sqrt( (x_estimate - x_true(i))^2 + (y_estimate - y_true(i))^2 );
	pf_x_err(i) = x_estimate - x_true(i);
    pf_y_err(i) = y_estimate - y_true(i);
    pf_th_err(i) = th_estimate - th_true(i);
    % Restrict theta error to -pi and pi
    while pf_th_err(i) < pi
        pf_th_err(i) = pf_th_err(i) + 2*pi;
    end
    while pf_th_err(i) > pi
        pf_th_err(i) = pf_th_err(i) - 2*pi;
    end

    % [TODO plot commands here; rem to add landmarks]
	plot(x_estimate,y_estimate,'o','color',green,'markersize',5,'linewidth',2);
    plot(x_true(i),y_true(i),'o','color',red,'markersize',5,'linewidth',2);

end
