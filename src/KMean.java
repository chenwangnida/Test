import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import com.google.common.collect.Ordering;

public class KMean {

	public static void main(String[] args) {

		// we have a list of our locations we want to cluster. create a
		List<Location> locations = new ArrayList<Location>();
		Random rand = new Random();

		// add some test points
		for (int i = 0; i < 500; i++) {
			double x = rand.nextDouble();
			double y = rand.nextDouble();

			locations.add(new Location(x, y));
		}

		List<LocationWrapper> clusterInput = new ArrayList<LocationWrapper>(locations.size());
		for (Location location : locations)
			clusterInput.add(new LocationWrapper(location));

		// initialize a new clustering algorithm.
		// we use KMeans++ with 3 clusters and 10000 iterations maximum.
		KMeansPlusPlusClusterer<LocationWrapper> clusterer = new KMeansPlusPlusClusterer<LocationWrapper>(3, 10000);
		List<CentroidCluster<LocationWrapper>> clusterResults = clusterer.cluster(clusterInput);

		// output the clusters
		for (int i = 0; i < clusterResults.size(); i++) {
			System.out.println("Cluster " + i);
			SimpleRegression regression = new SimpleRegression();
			double[][] data = new double[clusterResults.get(i).getPoints().size()][2];

			int j = 0;

			for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {
				data[j][0] = locationWrapper.getLocation().getX();
				data[j][1] = locationWrapper.getLocation().getY();
				j++;
			}

			regression.addData(data);

			for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {

				double y_pre = regression.getSlope() * locationWrapper.getLocation().getX() + regression.getIntercept();
				double diff = Math.pow((y_pre - locationWrapper.getLocation().getY()), 2);
				locationWrapper.setDiff(diff);

			}

			LocationWrapper min = Collections.min(clusterResults.get(i).getPoints(),
					Comparator.comparingDouble(a -> a.diff));

			// calculate distance to the most fit point
			for (LocationWrapper locationWrapper : clusterResults.get(i).getPoints()) {

				double euclidean_distance = calculateDistance(locationWrapper.getPoint(), min.getPoint());
				locationWrapper.setEuclidean_distance(euclidean_distance);
			}
		}

	}

	public static class LocationWrapper implements Clusterable {
		private double[] points;
		private Location location;
		private double diff;// difference between y and predicted y
		private double euclidean_distance;

		public LocationWrapper(Location location) {
			this.location = location;
			this.points = new double[] { location.getX(), location.getY() };
		}

		public Location getLocation() {
			return location;
		}

		public double[] getPoint() {
			return points;
		}

		public double getDiff() {
			return diff;
		}

		public void setDiff(double diff) {
			this.diff = diff;
		}

		public double getEuclidean_distance() {
			return euclidean_distance;
		}

		public void setEuclidean_distance(double euclidean_distance) {
			this.euclidean_distance = euclidean_distance;
		}

	}

	private static double calculateDistance(double[] vector1, double[] vector2) {
		double sum = 0;
		for (int i = 0; i < vector1.length; i++) {
			sum += Math.pow((vector1[i] - vector2[i]), 2);
		}
		return Math.sqrt(sum);
	}

}
