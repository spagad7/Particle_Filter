#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cfloat>
#include <limits>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 25;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	std::default_random_engine gen;

	for(int i = 0; i < num_particles; ++i) {
		struct Particle p;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	double vel_by_yaw = velocity / yaw_rate;
	double yaw_change = yaw_rate * delta_t;
	std::default_random_engine gen;
	
	// add measurements with noise to each particle
	for(int i = 0; i < num_particles; ++i) {
		// before adding noise
		double x_new, y_new, theta_new;

		if(std::abs(yaw_rate) < std::numeric_limits<double>::epsilon()) {
			x_new = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
			y_new = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
			theta_new = particles[i].theta;
		} else {
			x_new = particles[i].x + vel_by_yaw * (sin(particles[i].theta + yaw_change) - sin(particles[i].theta));
			y_new = particles[i].y + vel_by_yaw * (cos(particles[i].theta) - cos(particles[i].theta + yaw_change));
			theta_new = particles[i].theta + yaw_change;
		}

		// add noise by sampling from normal distribution
		normal_distribution<double> dist_x(x_new, std_pos[0]);
		normal_distribution<double> dist_y(y_new, std_pos[1]);
		normal_distribution<double> dist_theta(theta_new, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}


void ParticleFilter::getAssociatedLandmarks(std::vector<LandmarkObs>& obs_trans,
											std::vector<LandmarkObs>& matching_landmarks, const Map &map) {

	// for each transformed observation look for closest landmark (O(mn))
	for(int i = 0; i < (int)obs_trans.size(); ++i) {
		double o_x = obs_trans[i].x;
		double o_y = obs_trans[i].y;
		double dist_min = DBL_MAX;
		LandmarkObs l_nearest;

		for(int j = 0; j < (int)map.landmark_list.size(); ++j) {
			int m_id = map.landmark_list[j].id_i;
			double m_x = (double)map.landmark_list[j].x_f;
			double m_y = (double)map.landmark_list[j].y_f;
			double dist = sqrt(pow((o_x - m_x),2) + pow((o_y - m_y), 2));

			if(dist < dist_min) {
				l_nearest.id = m_id;
				l_nearest.x = m_x;
				l_nearest.y = m_y;
				dist_min = dist;
			}
		}

		matching_landmarks.push_back(l_nearest);
	}
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// calculate constant used in weight update
	// prevents unnecessary repeated calculation when put inside the loop
	double c = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	double d = 2 * std_landmark[0] * std_landmark[0];

	// clear weights vector before adding updated weights of resampled particles
	weights.clear();

	// update weight of each particle
	for(int i = 0; i < num_particles; ++i) {
		particles[i].weight = 1.0; // reset particle weight before updating it

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// transform observations from vehicle coordinate system to map coordinate system
		std::vector<LandmarkObs> obs_trans;
		for(int j = 0; j < (int)observations.size(); ++j) {
			LandmarkObs obs;
			obs.id = observations[j].id;
			obs.x = p_x + (observations[j].x * cos(p_theta)) - (observations[j].y * sin(p_theta));
			obs.y = p_y + (observations[j].x * sin(p_theta)) + (observations[j].y * cos(p_theta));
			obs_trans.push_back(obs);
		}

		// associate observations with nearest map landmarks
		std::vector<LandmarkObs> matching_landmarks;
		getAssociatedLandmarks(obs_trans, matching_landmarks, map_landmarks);

		// calculate weight of the particle
		for(int j = 0; j < (int)obs_trans.size(); ++j) {
			double e1 = pow((matching_landmarks[j].x - obs_trans[j].x),2) / d;
			double e2 = pow((matching_landmarks[j].y - obs_trans[j].y),2) / d;
			particles[i].weight =  particles[i].weight * c * exp(-(e1 + e2));
		}

		// store particle weight in weights vector for easy resampling
		weights.push_back(particles[i].weight);
	}
}


void ParticleFilter::resample() {

	// pick particles based on the probability of their weights
	discrete_distribution<> dist(weights.begin(), weights.end());
	std::default_random_engine gen;
	std::vector<Particle> particles_resampled;

	for(int i = 0; i < num_particles; ++i) {
		Particle p = particles[dist(gen)];
		particles_resampled.push_back(p);
	}

	particles = particles_resampled;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}


string ParticleFilter::getAssociations(Particle best) {
	
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}


string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
