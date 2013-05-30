import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class kmeans {
	public static void main(String args[]) {

		try {
			// Load the Data.
			DataSource source = new DataSource("../data/two_cluster.arff");
			Instances data = source.getDataSet();

			// Perform kmeans with 2 clusters.
			SimpleKMeans kmeans = new SimpleKMeans();
			kmeans.setPreserveInstancesOrder(true);
			kmeans.setNumClusters(2);
			kmeans.buildClusterer(data);

			// Show cluster association.
			int[] assignments = kmeans.getAssignments();
			for (int cluster : assignments)
				System.out.println(cluster);

			// Show cluster centers.
			Instances centers = kmeans.getClusterCentroids();
			System.out.println(centers);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
