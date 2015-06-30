#pragma once
#include <vector>
#include <queue>
#include <opencv/cv.h>
#include <memory>
#include <functional>
#include <cmath>
#include <boost/heap/fibonacci_heap.hpp>

using namespace std;
using namespace cv;
using namespace boost::heap;

struct comparePair
{
	bool operator() (const pair<float, int>& a, const pair<float, int>& b) const
	{
		return a.first > b.first;
	}
};

typedef fibonacci_heap< pair<float, int>, boost::heap::compare<comparePair>>::handle_type handle_t;

template <class T>
class OpticsClustering
{

public:
	OpticsClustering() :
		m_eps(0),
		m_minPoints(0),
		m_visited(),
		m_core(),
		m_reach(),
		m_queue(),
		m_queueHandles()
	{}

	virtual ~OpticsClustering(){};

	void clusterPoints(std::vector<T>* points,
		float eps, unsigned int minPoints,
		std::function<float(T, T)>getDistance)
	{
		m_visited.resize(points->size(), false);
		m_core.resize(points->size(), nanf(""));
		m_reach.resize(points->size(), nanf(""));
		m_queueHandles.resize(points->size(), handle_t());

		m_eps = eps;
		m_minPoints = minPoints;

		m_points = points;

		m_getDistance = getDistance;

		clusterPoints();
	};

	void getClusters(vector<int>& indices, vector<float>& distances)
	{
		indices = m_orderId;
		distances = m_reach;
	}



private:

	void getCoreDistance(int i, vector<int>& neighbors, vector<float>& distances)
	{
		if (neighbors.size() >= m_minPoints)
		{
			m_core[i] = m_eps;
			for (size_t i = 0; i < neighbors.size(); i++)
			{
				m_core[i] = min(m_core[i], distances[i]);
			}
		}
	}

	void clusterPoints()
	{
		int clusterId = 0;
		
		for (int i = 0; i < m_points->size(); i++)
		{
			if (m_visited[i])
				continue;
			// mark p as visited
			m_visited[i] = true;
			m_orderId.push_back(i);
			// get neighbors
			vector<int> neighbors;
			vector<float> neighDistances;
			regionQuery(i, neighbors, neighDistances);

			if (!isnan(m_core[i]))
			{
				update(i, neighbors, neighDistances);

				while (!m_queue.empty())
				{
					int next = m_queue.top().second;
					m_reach[next] = m_queue.top().first;
					m_queue.pop();

					vector<int> nNeighbors;
					vector<float> nNeighDistances;

					regionQuery(next, nNeighbors, nNeighDistances);

					m_visited[next] = true;
					m_orderId.push_back(next);

					if (!isnan(m_core[next]))
					{
						update(next, nNeighbors, nNeighDistances);
					}
				}
			}


		}
	};

	void regionQuery(int id, std::vector<int>& indices, vector<float>& distances)
	{
		for (int i = 0; i < m_points->size(); i++)
		{
			if (i != id)
			{
				float distance = m_getDistance(m_points->at(id), m_points->at(i));
				if (distance < m_eps)
				{
					indices.push_back(i);
					distances.push_back(distance);
				}
			}
		}
		getCoreDistance(id, indices, distances);
	};

	void update(int id, vector<int>& neighbors, vector<float>& distances)
	{
		float& cDist = m_core[id];

		for (size_t i = 0; i < neighbors.size(); i++)
		{
			int neighbor = neighbors[i];

			if (!m_visited[neighbor])
			{
				float reachDist = max(cDist, distances[i]);

				if (isnan(m_reach[neighbor]))
				{
					m_reach[neighbor] = reachDist;
					m_queueHandles[neighbor] = 
						m_queue.push(pair<float, int>(m_reach[neighbor], neighbor));
				}
				else if (reachDist < m_reach[neighbor])
				{
					//TODO: test increase function here, maybe it is better
					m_reach[neighbor] = reachDist;
					m_queue.update(
						m_queueHandles[neighbor], pair<float, int>(m_reach[neighbor], neighbor));
				}
			}
		}
	}

	float m_eps;
	unsigned int m_minPoints;

	vector<T>* m_points;
	vector<float> m_core;
	vector<float> m_reach;
	vector<bool> m_visited;
	vector<int> m_orderId;

	std::function<float(T, T)> m_getDistance;
 
	//priority_queue<pair<float, int>, std::vector<pair<float, int> >, 
	//	function<bool(pair<float, int>&, pair<float, int>&)> > m_queue;
	
	fibonacci_heap< pair<float, int>, boost::heap::compare<comparePair > > m_queue;
	vector<handle_t> m_queueHandles;

};

