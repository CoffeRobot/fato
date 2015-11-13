/*****************************************************************************/
/*  Copyright (c) 2015, Alessandro Pieropan                                  */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#ifndef DBSCAN_H
#define DBSCAN_H


#include<vector>
#include <memory>
#include <functional>

template <class T>
class DBScanClustering
{

public:
	DBScanClustering() :
		m_eps(0),
		m_minPoints(0),
		m_visited(),
		m_noise(),
		m_clustered(),
		m_clusters(){};

	virtual ~DBScanClustering(){};

	void clusterPoints(std::vector<T>* points, 
					   float eps, unsigned int minPoints,
					   std::function<float(T, T)>getDistance )
	{
		m_visited.resize(points->size(), false);
		m_noise.resize(points->size(), false);
		m_clustered.resize(points->size(), false);
		m_border.resize(points->size(), true);
		m_clusters.clear();

		m_eps = eps;
		m_minPoints = minPoints;

		m_points = points;

		m_getDistance = getDistance;

		clusterPoints();
	};

	std::vector<std::vector<int> > getClusters(){ return m_clusters; }

	void getBorderClusters(std::vector<std::vector<int> >& clusters, std::vector<bool>& borders)
	{
		clusters = m_clusters;
		borders = m_border;
	}



private:

	void clusterPoints()
	{
		int clusterId = 0;
        m_clusters.push_back(std::vector<int>());
		for (int i = 0; i < m_points->size(); i++)
		{
			if (m_visited[i])
				continue;

			m_visited[i] = true;

            std::vector<int> indices;

			regionQuery(i, indices);

			if (indices.size() < m_minPoints)
				m_noise[i] = true;
			else
			{
				expandCluster(i, indices, clusterId);
				clusterId++;
                m_clusters.push_back(std::vector<int>());
				m_border[i] = false;
			}

		}
	};

	void regionQuery(int id, std::vector<int>& indices)
	{
		for (int i = 0; i < m_points->size(); i++)
		{

			float distance = m_getDistance(m_points->at(id), m_points->at(i));
			if (distance < m_eps)
				indices.push_back(i);

		}
	};

	void expandCluster(int id, std::vector<int>& neighbors, int clusterId)
	{
		m_clusters[clusterId].push_back(id);
		m_clustered[id] = true;

		for (int i = 0; i < neighbors.size(); i++)
		{
			int ng = neighbors[i];

			if (!m_visited[ng])
			{
				m_visited[ng] = true;
                std::vector<int> newNeighbors;
				regionQuery(ng, newNeighbors);

				if (newNeighbors.size() >= m_minPoints)
				{
					neighbors.insert(neighbors.end(), newNeighbors.begin(), newNeighbors.end());
					m_border[ng] = false;
				}
					
			}
			if (!m_clustered[ng])
			{
				m_clusters[clusterId].push_back(ng);
				m_clustered[ng] = true;
			}
		}
	};

	float m_eps;

	unsigned int m_minPoints;

    std::vector<bool> m_noise;
    std::vector<bool> m_visited;
    std::vector<bool> m_clustered;
    std::vector<bool> m_border;

    std::vector<T>* m_points;
	
    std::vector<std::vector<int> > m_clusters;

	std::function<float(T,T)> m_getDistance;
	
//	float (*m_getDistance)(T, T);


};

#endif
