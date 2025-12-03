class SwitchParameters: public Parameters
{
public:
	SwitchParameters();
	~SwitchParameters() {};
	
	double prob_switch;
	virtual bool set(string &, string &);
	virtual bool arrange();
	virtual void usage();
};

SwitchParameters::SwitchParameters() : Parameters() 
{
	prob_switch = 0.1;

	command_flags.push_back("-p");		//14	
}

bool SwitchParameters::arrange() 
{
	if (prob_switch<0 || prob_switch > 1) 
	{
		cerr<<"\n***********************\nERROR:\t invalid switch probability"<<endl;
		return false;
	}
	
	if( !Parameters::arrange() )
	{
		return false;
	}
	cout<<"probability of switching:\t"<< prob_switch <<endl;
	
	return true;
}

bool SwitchParameters::set(string & flag, string & num) 
{
	// false is something goes wrong
	double err;
	if (!cast_string_to_double(num, err)) 
	{
		cerr<<"\n***********************\nERROR while reading parameters"<<endl;
		return false;
	}
	if (flag==command_flags[14]) 
	{
		prob_switch = err;
		return true;
	}
	return Parameters::set(flag,num);
}

void SwitchParameters::usage()
{
	Parameters::usage();
	cout<<"-p\t\t[probability of a node switching community membership between time steps]"<<endl;
}

// ------------------------------------------------------------------------------------------------------------

class SwitchGenerator: public BaseGenerator
{
public:
	SwitchGenerator( SwitchParameters* param );
	SwitchGenerator( Parameters* param, double prob );
	~SwitchGenerator() {};
	virtual int mutate( int step, int num_nodes, deque<deque<int> > &member_matrix, multimap<int,int> &last_step_dynamic_map, multimap<int,int> &step_dynamic_map );
	
protected:
	double prob_switch;
	void apply_switch( int num_nodes, deque<deque<int> > &member_matrix );
};

SwitchGenerator::SwitchGenerator(SwitchParameters* param) : BaseGenerator(param), prob_switch(param->prob_switch) 
{
	do_write_events = false;
} 

SwitchGenerator::SwitchGenerator(Parameters* param, double prob ): BaseGenerator(param), prob_switch(prob) {}

void SwitchGenerator::apply_switch( int num_nodes, deque<deque<int> > &member_matrix )
{
	int num_switch = (int)( num_nodes * prob_switch );
	printf("Switching community memberships (prob_switch=%.2f,num_switch=%d/%d)...\n", prob_switch, num_switch,num_nodes );

	int k = member_matrix.size();

	set<int> reassigned_nodes;
	for(int i = 0; i < num_switch; i++)
	{
		int from_cluster_index, to_cluster_index;
		int pos, node_index;
		// We reassign under the following conditions:
		// 1. From community is above min size
		// 2. Node hasn't already been assigned
		// 3. Not going into a community that it's already assigned to
		bool found_in_dest = false;
		int attempts = 0;
		bool failed = false;
		do
		{
			attempts++;
			if( attempts > 20000 )
			{
				failed = true;
				break;
			}
			// chose source
			from_cluster_index = irand( k - 1 );
			// choose destination
			to_cluster_index = irand( k - 1 );
			// choose node
			pos = irand(member_matrix[from_cluster_index].size()-1);
			node_index = member_matrix[from_cluster_index][pos];
			// is it already in the destination?
			found_in_dest = find(member_matrix[to_cluster_index].begin(), member_matrix[to_cluster_index].end(), node_index) != member_matrix[to_cluster_index].end();
		}
		while( from_cluster_index == to_cluster_index || found_in_dest || member_matrix[from_cluster_index].size() <= MIN_CLUSTER_SIZE+1 || reassigned_nodes.count( node_index ) );
		if( failed )
		{
			printf("Not able to perform any more switches\n");
			break;
		}
		// remove & add
		member_matrix[from_cluster_index].erase( member_matrix[from_cluster_index].begin() + pos );
		member_matrix[to_cluster_index].push_back( node_index );
		reassigned_nodes.insert(node_index);
#ifdef DEBUG_MUTATION
		printf("Switch: node %d C%d->C%d. Size C%d now %d\n", node_index, from_cluster_index+1, to_cluster_index+1, from_cluster_index+1, (int)member_matrix[from_cluster_index].size() );
#endif
	}
	printf("%d node memberships switched\n", (int)reassigned_nodes.size() );	
}

int SwitchGenerator::mutate( int step, int num_nodes, deque<deque<int> > &member_matrix, multimap<int,int> &last_step_dynamic_map, multimap<int,int> &step_dynamic_map )
{
	apply_switch( num_nodes, member_matrix );

	// simple step->dynamic mapping (i.e. just copy last)
	step_dynamic_map.clear();
	step_dynamic_map.insert(last_step_dynamic_map.begin(), last_step_dynamic_map.end());
	
	// reorder again, now that we've changed
	for (int i=0; i<member_matrix.size(); i++)
	{
		sort(member_matrix[i].begin(), member_matrix[i].end());
	}
	return 0;
}

